
#include "CSLM.h"
#include "moses/FactorCollection.h"
#include "PointerState.h"
#include <iostream>
//#include <boost/python/suite/indexing/map_indexing_suite.hpp> // Needed?
//#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

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
  
  }
  
  void CSLM::Load() {
    // Note that Boost does not support Py_Finalize();
  }
  
  void CSLM::LoadThread() {
    // This gets called once for each thread, before forking
    thread_id = boost::lexical_cast<std::string>(boost::this_thread::get_id());
    memory_id = thread_id + "memory";
    
    // Create the messaging queue
//    std::string thread_id = boost::lexical_cast<std::string>(boost::this_thread::get_id());
//    mq.reset(new boost::interprocess::message_queue(boost::interprocess::open_or_create, thread_id.c_str(), 1, sizeof(int)));
    
//    std::string thread_id = boost::lexical_cast<std::string>(boost::this_thread::get_id());
//    boost::interprocess::message_queue::remove(thread_id.c_str());
//    boost::interprocess::message_queue mq(boost::interprocess::create_only, thread_id.c_str(), 1, sizeof(int));
//    std::cout << "Created message queue with name: " << thread_id << std::flush << std::endl;
    
    // Create the shared memory segment
//    boost::interprocess::shared_memory_object::remove(thread_id.c_str());
//    boost::interprocess::managed_shared_memory segment(boost::interprocess::create_only, thread_id.c_str(), 65536);
//    VoidAllocator alloc_inst(segment.get_segment_manager());
//    MapType *requests = segment.construct<MapType>("MyMap")
//      (std::less<IntVector>(), alloc_inst);
  }
  
  void CSLM::LoadChild() {
    // This gets called once in the child process
//    std::cout << "Loading CSLM child process.." << std::flush << std::endl;
    
    // Start Python
    std::cout << "Starting Python..." << std::flush << std::endl;
//    std::cout << Py_GetProgramFullPath() << std::endl;
//    std::cout << Py_IsInitialized() << std::endl;
//    Py_Finalize();
//    PyEval_InitThreads();
//    std::cout << Py_GetVersion() << std::endl;
//    try {
//    PyInterpreterState_New();
//    Py_Initialize();
//    PyEval_InitThreads();
//    } catch (boost::python::error_already_set) {
//      PyErr_Print();
//    }
    std::cout << "Started Python" << std::flush << std::endl;
    // Exposing these gives an error right now
//    boost::python::class_<MapType>("ContextFactors")
//    .def(boost::python::map_indexing_suite<MapType>());
//    boost::python::class_<IntVector>("Phrase")
//    .def(boost::python::vector_indexing_suite<IntVector>());
//    try {
//      std::cout << "Loading Python module" << std::flush << std::endl;
//      boost::python::object py_cslm = boost::python::import("cslm");
//      py_run_cslm = py_cslm.attr("run_cslm");
//    } catch(boost::python::error_already_set const &) {
//      PyErr_Print();
//    }
    
    boost::interprocess::shared_memory_object::remove(memory_id.c_str());
    boost::interprocess::managed_shared_memory segment(boost::interprocess::create_only, memory_id.c_str(), 65536);
    VoidAllocator alloc_inst(segment.get_segment_manager());
    MapType *requests = segment.construct<MapType>("MyMap")
      (std::less<IntVector>(), alloc_inst);
//    boost::interprocess::interprocess_mutex *mtx = segment.find_or_construct<boost::interprocess::interprocess_mutex>("mtx")();
//    mtx->unlock();

    
    // Create the messaging queue
//    std::cout << "Opening the message queue..." << std::flush << std::endl;
//    std::string thread_id = boost::lexical_cast<std::string>(boost::this_thread::get_id());
//    boost::interprocess::message_queue mq(boost::interprocess::open_only, thread_id.c_str());
//    std::string thread_id = boost::lexical_cast<std::string>(boost::this_thread::get_id());
    boost::interprocess::message_queue::remove(thread_id.c_str());
    boost::interprocess::message_queue mq(boost::interprocess::create_only, thread_id.c_str(), 1, sizeof(int));
    std::cout << "Created message queue with name: " << thread_id << std::flush << std::endl;
    
    // Open the shared memory segment
//    boost::interprocess::managed_shared_memory segment(boost::interprocess::open_only, thread_id.c_str());
//    MapType *requests = segment.find<MapType>("MyMap").first;
    
    // Ready to go! Start listening to messages
    int message;
    unsigned int priority;
    boost::interprocess::message_queue::size_type recvd_size;
    Py_Initialize();
    while (true) {
//      std::cout << "Waiting for message from Moses" << std::flush << std::endl;
      mq.receive(&message, sizeof(message), recvd_size, priority);
//      std::cout << "RECEIVED " << message << std::endl;
      if (message != 1 || recvd_size != sizeof(message)) {
        std::cout << "Error in message to Python" << std::endl;
      } else {
        // We got a message that a batch is ready; process it
//        std::cout << "Requests size: " << requests->size() << std::endl;
        try {
          RunPython();
//          boost::python::object scores = py_run_cslm(requests);
        } catch(boost::python::error_already_set const &) {
          PyErr_Print();
        }
      }
      if (getppid() == 1) {
        std::cout << "Quiting child process" << std::endl;
        exit(0);
      }
    }
  }
  
  void CSLM::RunPython() {
    std::cout << "Running Python!" << std::endl;
  }
  
  void CSLM::SetFFStateIdx(int state_idx) {
    m_state_idx = state_idx;
  }
  
  void CSLM::IssuePythonRequest(std::vector<const Word*> contextFactor) {
    // Access memory segment, create allocator, open shared map, insert...
    // Most of these things really shouldn't be done each time!
//    std::string thread_id = boost::lexical_cast<std::string>(boost::this_thread::get_id());
    boost::interprocess::managed_shared_memory segment(boost::interprocess::open_only, memory_id.c_str());
//    boost::interprocess::interprocess_mutex *mtx = segment.find_or_construct<boost::interprocess::interprocess_mutex>("mtx")();
//    mtx->lock();
    VoidAllocator alloc_inst(segment.get_segment_manager());
//    std::cout << requests->size() << std::endl;
//    std::cout << segment.get_num_named_objects() << std::endl;
//    std::cout << segment.get_size() << std::endl;
    IntVector phrase(alloc_inst);
//    IntVector *phrase = segment.construct<IntVector>("phrase")
//      (alloc_inst);
//    std::cout << "Adding an n-gram" << std::endl;
    for (int i = 0; i < contextFactor.size(); i++) {
      phrase.push_back(contextFactor[i]->GetFactor(0)->GetId());
    }
    MapType *requests = segment.find<MapType>("MyMap").first;
    requests->insert(MapElementType(phrase, 0.0));
//    mtx->unlock();
  }
  
  void CSLM::IssueRequestsFor(Hypothesis& hypo, const FFState* input_state) {
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
//    std::vector<std::string> phrase;
//    for (int i = 0; i < contextFactor.size(); i++) {
//      phrase.push_back(contextFactor[i]->GetString(0).as_string());
//    }
//    std::map<std::vector<std::string>, float>::const_iterator map_lookup = requests.find(phrase);
//    if (map_lookup == requests.end()) {
//      // Throw an error!
//      cout << "ERROR" << endl;
//    } else {
//      ret.score = map_lookup->second;
//    }
//    ret.score = requests.contextFactor];
    ret.score = 0.0;
    ret.unknown = false;
    
    // Get scores from sync here
    
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
  
  void CSLM::SyncBuffer() {
//    boost::interprocess::managed_shared_memory segment(boost::interprocess::open_only, thread_id.c_str());
//    MapType *requests = segment.find<MapType>("MyMap").first;
//    if (requests->size() > 0) {
//      int message = 1;
//    unsigned int priority;
//    boost::interprocess::message_queue::size_type recvd_size;
//    mq->receive(&message, sizeof(message), recvd_size, priority);
//    }
//    std::string thread_id = boost::lexical_cast<std::string>(boost::this_thread::get_id());
    boost::interprocess::message_queue mq(boost::interprocess::open_only, thread_id.c_str());
    int message = 1;
    mq.send(&message, sizeof(int), 0);
  }
  
  void CSLM::ClearBuffer() {
//    std::string thread_id = boost::lexical_cast<std::string>(boost::this_thread::get_id());
//    boost::interprocess::managed_shared_memory segment(boost::interprocess::open_only, thread_id.c_str());
//    MapType *requests = segment.find<MapType>("MyMap").first;
//    requests->clear();
  }
  
}



