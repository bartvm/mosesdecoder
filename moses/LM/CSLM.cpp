
#include "CSLM.h"
#include "moses/FactorCollection.h"
#include "PointerState.h"
#include <iostream>
#include <boost/filesystem.hpp>

namespace Moses {
  CSLM::CSLM(const std::string &line) : LanguageModelSingleFactor(line) {
    ReadParameters();
    
    FactorCollection &factorCollection = FactorCollection::Instance();
    
    // Needed by parent language model classes
    m_sentenceStart = factorCollection.AddFactor(Output, m_factorType, BOS_);
    m_sentenceStartWord[m_factorType] = m_sentenceStart;
    
    m_sentenceEnd		= factorCollection.AddFactor(Output, m_factorType, EOS_);
    m_sentenceEndWord[m_factorType] = m_sentenceEnd;
  }
  
  CSLM::~CSLM() {
    boost::interprocess::message_queue moses_to_py(boost::interprocess::open_only, ThisThreadId("from").c_str());
    int message = 2;
    moses_to_py.send(&message, sizeof(int), 0);
    boost::interprocess::message_queue::remove(ThisThreadId("from").c_str());
  }
  
  void CSLM::Load() {}
  
  std::string CSLM::ThisThreadId(std::string prefix) const {
    // For reasons beyond my comprehension, storing the thread ID in a
    // variable doesn't work; multiple threads end up getting the same ID
    // and the messaging queues/shared memory crash. Requesting the thread ID
    // through a function like this seems to work
    return prefix + boost::lexical_cast<std::string>(boost::this_thread::get_id());
  }
  
  void CSLM::LoadThread() {
    // This gets called once for each thread at the very beginning
    
    // Setting up message queues
    // 0 signals READY
    // 1 signals NEXT
    // 2 signals EXIT
    // other ERROR
    // Delete old messaging queues; should be unnecessary, but you never know
    boost::interprocess::message_queue::remove(ThisThreadId("from").c_str());
    boost::interprocess::message_queue::remove(ThisThreadId("to").c_str());
    // Create new queues
    boost::interprocess::message_queue py_to_moses(boost::interprocess::create_only, ThisThreadId("to").c_str(), 1, sizeof(int));
    boost::interprocess::message_queue moses_to_py(boost::interprocess::create_only, ThisThreadId("from").c_str(), 1, sizeof(int));
    
    // Setting up the managed shared memory segment; first remove old one (just in case)
    boost::interprocess::shared_memory_object::remove(ThisThreadId("memory").c_str());
    boost::interprocess::managed_shared_memory segment(boost::interprocess::create_only, ThisThreadId("memory").c_str(), 65536);
    stldb::scoped_allocation<segment_manager_t> scope(segment.get_segment_manager());
    MapType *requests = segment.construct<MapType>("MyMap")
      (std::less<IntVector>());
    
    // Create the PyMoses command to execute and pipe PyMoses's stdout back to the parent
    FILE *fpipe;
    boost::filesystem::path cwd(boost::filesystem::current_path());
    std::string pymoses_path = cwd.string() + "/bin/pymoses";
    std::string command = pymoses_path + " " + ThisThreadId("");
    char line[256];
    
    // Fork the current process, message PyMoses in the child process, don't continue until child process has answered
    int pid = fork();
    if (pid > 0) {
      // PARENT PROCESS; We wait for the child to signal okay
      std::cout << "Waiting for OK sign from process " << pid << ", child of thread " << ThisThreadId("pre") << "... " << std::flush;
      int message = 1;
      boost::interprocess::message_queue::size_type recvd_size;
      unsigned int priority;
      py_to_moses.receive(&message, sizeof(message), recvd_size, priority);
      if (message == 0) {
        std::cout << "OK!" << std::endl;
      } else {
        std::cout << "PyMoses sent bad message!" << std::endl;
        exit(1);
      }
    } else if (pid == 0) {
      // CHILD PROCESS; Start pymoses and pipe the results back
      if (!(fpipe = (FILE*)popen(command.c_str(), "r"))) {
        std::cout << "Problems with pipe" << std::endl;
        exit(1);
      }
      while (fgets( line, sizeof line, fpipe)) {
        printf("%s", line);
      }
      pclose(fpipe);
    } else {
      std::cout << "Forking error" << std::endl;
    }
  }
    
  void CSLM::SetFFStateIdx(int state_idx) {
    m_state_idx = state_idx;
  }
  
  void CSLM::IssuePythonRequest(std::vector<const Word*> contextFactor) {
    // Access memory segment, create allocator, open shared map, insert n-gram
    // Most of these things really shouldn't be done each time! See if there
    // is a more efficient way
    boost::interprocess::managed_shared_memory segment(boost::interprocess::open_only, ThisThreadId("memory").c_str());
    stldb::scoped_allocation<segment_manager_t> scope(segment.get_segment_manager());
    
    // Create the n-gram from factor IDs (should change to strings)
    IntVector phrase; // Note: Using the segment.construct() command doesn't work
    for (int i = 0; i < contextFactor.size(); i++) {
      phrase.push_back(contextFactor[i]->GetFactor(0)->GetId());
    }
    MapType *requests = segment.find<MapType>("MyMap").first;
    // Insert the n-gram with a placeholder score of 0.0
    requests->insert(MapElementType(phrase, 0.0));
  }
  
  void CSLM::IssueRequestsFor(Hypothesis& hypo, const FFState* input_state) {
    // This is called for each hypothesis; we construct all the possible
    // n-grams for this phrase and issue a scoring request to Python
    if(GetNGramOrder() <= 1) {
      return;
    }
    // Empty phrase added? Nothing to be done
    if (hypo.GetCurrTargetLength() == 0) {
      return;
    }
    
    const size_t currEndPos = hypo.GetCurrTargetWordsRange().GetEndPos();
    const size_t startPos = hypo.GetCurrTargetWordsRange().GetStartPos();
    
    // First n-gram
    std::vector<const Word*> contextFactor(GetNGramOrder());
    size_t index = 0;
    for (int currPos = (int) startPos - (int) GetNGramOrder() + 1;
         currPos <= (int) startPos; currPos++) {
      if (currPos >= 0) {
        contextFactor[index++] = &hypo.GetWord(currPos);
      } else {
        contextFactor[index++] = &GetSentenceStartWord();
      }
    }
    IssuePythonRequest(contextFactor);

    // Main loop
    size_t endPos = std::min(startPos + GetNGramOrder() - 2, currEndPos);
    for (size_t currPos = startPos + 1 ; currPos <= endPos ; currPos++) {
      // Shift all args down 1 place
      for (size_t i = 0 ; i < GetNGramOrder() - 1 ; i++) {
        contextFactor[i] = contextFactor[i + 1];
      }
      // Add last factor
      contextFactor.back() = &hypo.GetWord(currPos);
      IssuePythonRequest(contextFactor);
    }

    // End of sentence
    if (hypo.IsSourceCompleted()) {
      const size_t size = hypo.GetSize();
      contextFactor.back() = &GetSentenceEndWord();
      for (size_t i = 0 ; i < GetNGramOrder() - 1 ; i ++) {
        int currPos = (int)(size - GetNGramOrder() + i + 1);
        if (currPos < 0) {
          contextFactor[i] = &GetSentenceStartWord();
        } else {
          contextFactor[i] = &hypo.GetWord((size_t)currPos);
        }
      }
      IssuePythonRequest(contextFactor);
    } else {
      if (endPos < currEndPos) {
        // Need to get the LM state (otherwise the last LM state is fine)
        for (size_t currPos = endPos+1; currPos <= currEndPos; currPos++) {
          for (size_t i = 0 ; i < GetNGramOrder() - 1 ; i++) {
            contextFactor[i] = contextFactor[i + 1];
          }
          contextFactor.back() = &hypo.GetWord(currPos);
        }
      }
    }
  }
  
  LMResult CSLM::GetValue(const std::vector<const Word*> &contextFactor,
                          State* finalState) const {
    LMResult ret;
    
    // Access the memory, create an n-gram again from the context factor
    // and find the score in the shared memory. NOTE: This duplicates the
    // n-gram creation from when the scoring request was issued; probably
    // should just save the std::vector<const Word*> and IntVector in
    // an (unordered) map?
    boost::interprocess::managed_shared_memory segment(boost::interprocess::open_only, ThisThreadId("memory").c_str());
    stldb::scoped_allocation<segment_manager_t> scope(segment.get_segment_manager());
    IntVector phrase;
    for (int i = 0; i < contextFactor.size(); i++) {
      phrase.push_back(contextFactor[i]->GetFactor(0)->GetId());
    }
    MapType *requests = segment.find<MapType>("MyMap").first;
    ret.score = requests->at(phrase);
    ret.unknown = false;
    
    // Use last word as state info
    const Factor *factor;
    size_t hash_value(const Factor &f);
    if (contextFactor.size()) {
      factor = contextFactor.back()->GetFactor(m_factorType);
    } else {
      factor = NULL;
    }
    
    (*finalState) = (State*) factor;
    
    return ret;
  }
  
  void CSLM::SendBuffer() {
    // Here we message the child process that a complete batch has been stored
    // in the shared memory, and is ready to be scored
    boost::interprocess::message_queue moses_to_py(boost::interprocess::open_only, ThisThreadId("from").c_str());
    int message = 1;
    moses_to_py.send(&message, sizeof(int), 0);
  }
  
  void CSLM::SyncBuffer() {
    // Here we wait for the child process to finish scoring before we read out
    // the scores from shared memory
    boost::interprocess::message_queue py_to_moses(boost::interprocess::open_only, ThisThreadId("to").c_str());
    int message;
    boost::interprocess::message_queue::size_type recvd_size;
    unsigned int priority;
    py_to_moses.receive(&message, sizeof(message), recvd_size, priority);
    if (message != 1) {
      std::cout << "Received wrong message from PyMoses while waiting for eval" << std::endl;
      exit(1);
    }
  }
  
  void CSLM::ClearBuffer() {
    // All the hypotheses in this batch have been scored, so delete them from
    // the shared memory
    boost::interprocess::managed_shared_memory segment(boost::interprocess::open_only, ThisThreadId("memory").c_str());
    stldb::scoped_allocation<segment_manager_t> scope(segment.get_segment_manager());
    MapType *requests = segment.find<MapType>("MyMap").first;
    requests->clear();
  }
  
}



