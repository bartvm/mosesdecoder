
#include "CSLM.h"
#include "moses/FactorCollection.h"
#include "moses/StaticData.h"
#include "moses/Util.h"

using namespace std;

namespace Moses {
  CSLM::CSLM(const std::string &line)
    : LanguageModelSingleFactor(line) {
    ReadParameters();

    FactorCollection &factorCollection = FactorCollection::Instance();

    // needed by parent language model classes. Why didn't they set these themselves?
    m_sentenceStart = factorCollection.AddFactor(Output, m_factorType, BOS_);
    m_sentenceStart_CSLM = factorCollection.AddFactor(Output, 1, "0");
    m_sentenceStartWord[m_factorType] = m_sentenceStart;
    m_sentenceStartWord[1] = m_sentenceStart_CSLM;

    m_sentenceEnd		= factorCollection.AddFactor(Output, m_factorType, EOS_);
    m_sentenceEnd_CSLM = factorCollection.AddFactor(Output, 1, "0");
    m_sentenceEndWord[m_factorType] = m_sentenceEnd;
    m_sentenceEndWord[1] = m_sentenceEnd_CSLM;
  }

  CSLM::~CSLM() {}

  std::string CSLM::ThisThreadId(std::string prefix) const {
    // For reasons beyond my comprehension, storing the thread ID in a
    // variable doesn't work; multiple threads end up getting the same ID
    // and the messaging queues/shared memory crash. Requesting the thread ID
    // through a function like this seems to work (note: using a
    // thread_shared_ptr isn't any better and results in double free,
    // corruption, wrong pointers, and more!)
    std::stringstream ss;
    ss << prefix << pthread_self();
    return ss.str();
  }

  void CSLM::StopThread() {
    // Clean up message queues
    VERBOSE(1, "Removing message queues" << endl);
    int message = 2;
    moses_to_py->send(&message, sizeof(int), 0);
    boost::interprocess::message_queue::remove(ThisThreadId("to").c_str());

    // Clean up shared memory
    VERBOSE(1, "Removing shared memory" << endl);
    boost::interprocess::shared_memory_object::remove(
      ThisThreadId("memory").c_str()
    );
  }

  void CSLM::LoadThread() {
    // Getting a random ID for this thread
    // boost::random::random_device rng;
    // thread_id.reset((unsigned int*)rng());

    // Setting up message queues
    // 0 signals READY
    // 1 signals NEXT
    // 2 signals EXIT
    // other ERROR
    VERBOSE(1, "Setting up message queues" << endl);
    py_to_moses.reset(new boost::interprocess::message_queue(
      boost::interprocess::create_only, ThisThreadId("to").c_str(),
      1, sizeof(int))
    );
    moses_to_py.reset(new boost::interprocess::message_queue(
      boost::interprocess::create_only, ThisThreadId("from").c_str(),
      1, sizeof(int))
    );

    // Setting up the managed shared memory segment
    VERBOSE(1, "Setting up shared memory" << endl);
    segment.reset(new boost::interprocess::managed_shared_memory(
      boost::interprocess::create_only, ThisThreadId("memory").c_str(),
      419242304
    ));
    stldb::scoped_allocation<segment_manager_t> scope(segment->get_segment_manager());
    segment->construct<MapType>("MyMap")(std::less<IntVector>());

    // Create the PyMoses command to execute and pipe PyMoses's stdout back to the parent
    FILE *fpipe;
    boost::filesystem::path cwd(boost::filesystem::current_path());
    std::string pymoses_path = cwd.string() + "/bin/pymoses";
    std::string command = pymoses_path + " " + ThisThreadId("");
    char line[256];

    // Fork the current process, message PyMoses in the child process,
    // don't continue until child process has answered
    int pid = fork();
    if (pid > 0) {
      // PARENT PROCESS; We wait for the child to signal okay
      VERBOSE(1, "Waiting for OK sign from process " << pid << ", child of thread " << ThisThreadId("") << endl);
      int message = 1;
      boost::interprocess::message_queue::size_type recvd_size;
      unsigned int priority;
      py_to_moses->receive(&message, sizeof(message), recvd_size, priority);
      if (message == 0) {
        VERBOSE(1, "Received OK from process" << pid << endl);
        // And continue execution
      } else {
        VERBOSE(1, "PyMoses sent bad message!" << endl);
        StopThread();
        exit(1);
      }
    } else if (pid == 0) {
      // CHILD PROCESS; Start pymoses and pipe the results back
      if (!(fpipe = (FILE*)popen(command.c_str(), "r"))) {
        VERBOSE(1, "Problems with pipe" << endl);
        StopThread();
        exit(1);
      }
      while (fgets( line, sizeof line, fpipe)) {
        printf("%s", line);
      }
      pclose(fpipe);
    } else {
      VERBOSE(1, "Forking error" << endl);
      StopThread();
      exit(1);
    }
  }

  void CSLM::IssuePythonRequest(std::vector<const Word*> contextFactor) {
    stldb::scoped_allocation<segment_manager_t> scope(
      segment->get_segment_manager()
    );

    // Create the n-gram from factor IDs
    // Note: Using the segment.construct() command doesn't work
    IntVector phrase;
    for (unsigned int i = 0; i < contextFactor.size(); i++) {
      if(contextFactor[i]->GetFactor(1)) {
        try {
          int cslm_id = boost::lexical_cast<int>(
            contextFactor[i]->GetString(1).as_string()
          );
          phrase.push_back(cslm_id);
        } catch (const boost::bad_lexical_cast& e) {
          phrase.push_back(1);
          VERBOSE(1, "Python error: Got non-integer CSLM ID, defaulted to 1 ('"
                      << contextFactor[i]->GetString(1).as_string()
                      << "' for word '"
                      << contextFactor[i]->GetString(0).as_string() << "')");
        }
      } else {
        phrase.push_back(1);
      }
    }
    MapType *requests = segment->find<MapType>("MyMap").first;
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
    stldb::scoped_allocation<segment_manager_t> scope(
      segment->get_segment_manager()
    );
    IntVector phrase;
    for (unsigned int i = 0; i < contextFactor.size(); i++) {
      if(contextFactor[i]->GetFactor(1)) {
        int cslm_id = boost::lexical_cast<int>(
          contextFactor[i]->GetString(1).as_string()
        );
        phrase.push_back(cslm_id);
      } else {
        phrase.push_back(1);
      }
    }
    MapType *requests = segment->find<MapType>("MyMap").first;
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
    int message = 1;
    moses_to_py->send(&message, sizeof(int), 0);
  }

  void CSLM::SyncBuffer() {
    // Here we wait for the child process to finish scoring before we read out
    // the scores from shared memory
    int message = 0;
    boost::interprocess::message_queue::size_type recvd_size;
    unsigned int priority;
    py_to_moses->receive(&message, sizeof(message), recvd_size, priority);
    if (message != 1) {
      VERBOSE(1, "Received wrong message from PyMoses while waiting for eval"
                 << endl);
      exit(1);
    } else {
      VERBOSE(1, "Python scoring completed" << endl);
    }
  }

  void CSLM::ClearBuffer() {
    // All the hypotheses in this batch have been scored, so delete them from
    // the shared memory
    stldb::scoped_allocation<segment_manager_t> scope(
      segment->get_segment_manager()
    );
    MapType *requests = segment->find<MapType>("MyMap").first;
    requests->clear();
  }

}

