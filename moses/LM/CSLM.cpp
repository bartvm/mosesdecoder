#include "CSLM.h"
#include <sys/wait.h>

using namespace std;
using namespace boost::interprocess;
using namespace boost::posix_time;

namespace Moses {
  CSLM::CSLM(const std::string &line)
    : LanguageModelSingleFactor(line),
      ngrams(&NpyIterCleanup), scores(&NpyIterCleanup),source(&NpyIterCleanup),
      source_sentence(&InputTypeCleanup),
      conditional(false), backoff(false) {
    ReadParameters();

    FactorCollection &factorCollection = FactorCollection::Instance();

    // needed by parent language model classes. Why didn't they set these themselves?
    m_sentenceStart = factorCollection.AddFactor(Output, m_factorType, BOS_);
    m_sentenceStart_CSLM = factorCollection.AddFactor(Output, 1, "0");
    m_sentenceStartWord[m_factorType] = m_sentenceStart;
    m_sentenceStartWord[1] = m_sentenceStart_CSLM;

    m_sentenceEnd	= factorCollection.AddFactor(Output, m_factorType, EOS_);
    m_sentenceEnd_CSLM = factorCollection.AddFactor(Output, 1, "0");
    m_sentenceEndWord[m_factorType] = m_sentenceEnd;
    m_sentenceEndWord[1] = m_sentenceEnd_CSLM;
  }

  CSLM::~CSLM() {
    PyEval_RestoreThread(state);
    Py_Finalize();
  }

  void CSLM::Load() {
    Py_Initialize();
    VERBOSE(1, "Initialized Python" << endl);
    int ret = _import_array();
    UTIL_THROW_IF2(ret < 0, "Unable to load NumPy. Please make sure that you "
                            "are loading the same NumPy that was used "
                            "during compilation (" << NPY_VERSION << ")");
    VERBOSE(1, "Initialized NumPy" << endl);
    // Save the current thread state, release implicit GIL
    state = PyEval_SaveThread();
  }

  string CSLM::ThisThreadId(string suffix) const {
    // For reasons I do not understand I have been unable to
    // store the thread ID in a thread specific pointer
    std::stringstream ss;
    ss << boost::this_thread::get_id() << suffix;
    return ss.str();
  }

  void CSLM::StopThread() {
    // Delete the message queues as soon as PyMoses responds
    int message = -1;
    VERBOSE(1, "Sending termination message to PyMoses" << endl);
    m2py_tsp->send(&message, sizeof(int), 0);
    message_queue::size_type recvd_size;
    unsigned int priority;
    // Set &message to NULL?
    // Make this a time receive?
    VERBOSE(1, "Waiting for PyMoses to exit before cleaning up" << endl);
    ptime timeout = second_clock::universal_time() + seconds(10);
    if(!py2m_tsp->timed_receive(&message, sizeof(message), recvd_size,
                                priority, timeout)) {
      VERBOSE(1, "Did not receive a reply from PyMoses... Exiting anyway." << endl);
    } else {
      VERBOSE(1, "Received reply. Removing message queues and memory." << endl);
    }
    Cleanup();
  }

  void CSLM::SetParameter(const std::string& key, const std::string& value) {
    if (key == "conditional") {
      conditional = true;
      VERBOSE(1, "CSLM is running in conditional mode" << endl);
    } else if (key == "backoff") {
      backoff = true;
    } else {
      LanguageModelSingleFactor::SetParameter(key, value);
    }
  }

  void CSLM::Cleanup() {
    // Remove message queues
    message_queue::remove(ThisThreadId("m2py").c_str());
    message_queue::remove(ThisThreadId("py2m").c_str());

    // Clean up shared memory
    shared_memory_object::remove(
      ThisThreadId("ngrams").c_str()
    );
    shared_memory_object::remove(
      ThisThreadId("scores").c_str()
    );
    if (conditional) {
      shared_memory_object::remove(
        ThisThreadId("source").c_str()
      );
    }
  }

  void CSLM::LoadThread() {
    // Setting up message queues
    VERBOSE(1, "Setting up message queues" << endl);

    m2py_tsp.reset(new message_queue(open_or_create,
                                     ThisThreadId("m2py").c_str(), 1, sizeof(int)));
    py2m_tsp.reset(new message_queue(open_or_create,
                                     ThisThreadId("py2m").c_str(), 1, sizeof(int)));

    // Setting up the shared memory segment
    // Easier doing it this way than in a separate function
    // because some things seem to disappear in the stack
    VERBOSE(1, "Setting up shared memory" << endl);
    shared_memory_object ngrams_shm_obj(open_or_create,
                                        ThisThreadId("ngrams").c_str(),
                                        read_write);
    shared_memory_object scores_shm_obj(open_or_create,
                                        ThisThreadId("scores").c_str(),
                                        read_write);
    shared_memory_object* source_shm_obj;
    if (conditional) {
      source_shm_obj = new shared_memory_object(open_or_create,
                                                ThisThreadId("source").c_str(),
                                                read_write);
    }
    // Dereferencing a NULL pointer is undefined behaviour, so we'll
    // do with the pointer to the shared memory object here

    // TODO: These numbers should be calculated more precisely (now 1MB for each)
    ngrams_shm_obj.truncate(2097152);
    scores_shm_obj.truncate(1048576);
    if (conditional) {
      source_shm_obj->truncate(1048576);
    }
    ngrams_region_tsp.reset(new mapped_region(ngrams_shm_obj, read_write));
    scores_region_tsp.reset(new mapped_region(scores_shm_obj, read_write));
    if (conditional) {
      source_region_tsp.reset(new mapped_region(*source_shm_obj, read_write));
    }
    memset(ngrams_region_tsp->get_address(), 1, ngrams_region_tsp->get_size());
    memset(scores_region_tsp->get_address(), 1, scores_region_tsp->get_size());
    if (conditional) {
      memset(source_region_tsp->get_address(), 1, source_region_tsp->get_size());
    }

    // Create the Python NumPy wrappers and store iterators over them
    // We need the GIL for this!
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    int ngrams_nd = 2;
    npy_intp ngrams_dims[2] = {25000, m_nGramOrder};
    PyObject* ngrams_array = PyArray_SimpleNewFromData(ngrams_nd, ngrams_dims,
                                                       NPY_INT,
                                                       ngrams_region_tsp->get_address());
    NpyIter *ngrams_iter = NpyIter_New((PyArrayObject*)ngrams_array,
                                       NPY_ITER_READWRITE, NPY_KEEPORDER,
                                       NPY_NO_CASTING, NULL);
    Py_DECREF(ngrams_array);
    ngrams.reset(ngrams_iter);

    int scores_nd = 1;
    npy_intp scores_dims[1] = {25000};
    PyObject* scores_array = PyArray_SimpleNewFromData(scores_nd, scores_dims,
                                                       NPY_FLOAT,
                                                       scores_region_tsp->get_address());
    NpyIter *scores_iter = NpyIter_New((PyArrayObject*)scores_array,
                                       NPY_ITER_READWRITE, NPY_KEEPORDER,
                                       NPY_NO_CASTING, NULL);
    Py_DECREF(scores_array);
    scores.reset(scores_iter);

    if (conditional) {
      int source_nd = 1;
      npy_intp source_dims[1] = {250};
      PyObject* source_array = PyArray_SimpleNewFromData(source_nd, source_dims,
                                                         NPY_INT,
                                                         source_region_tsp->get_address());
      NpyIter *source_iter = NpyIter_New((PyArrayObject*)source_array,
                                         NPY_ITER_READWRITE, NPY_KEEPORDER,
                                         NPY_NO_CASTING, NULL);
      Py_DECREF(source_array);
      source.reset(source_iter);
    }

    PyGILState_Release(gstate);

    batch_count.reset(new int(0));

    // Create the PyMoses command to execute and pipe PyMoses's stdout back
    // to the parent
    FILE *fpipe;
    std::string command = "pymoses " + ThisThreadId("");
    if (conditional) {
      command = command + " conditional";
    }
    char line[256];

    // Fork the current process, message PyMoses in the child process,
    // don't continue until child process has answered
    int pid = fork();
    if (pid > 0) {
      // PARENT PROCESS; We wait for the child to signal it's alive
      int message = 1;
      message_queue::size_type recvd_size;
      unsigned int priority;
      VERBOSE(1, "Waiting for ALIVE sign for 30 s from process " << pid
                 << ", child of thread " << ThisThreadId("") << endl);
      ptime timeout = second_clock::universal_time() + seconds(30);
      if(!py2m_tsp->timed_receive(&message, sizeof(message), recvd_size,
                                  priority, timeout)) {
        Cleanup();
        UTIL_THROW(util::Exception, "No signal from PyMoses.");
      }
      // Now wait for PyMoses to load the models and compile the functions
      VERBOSE(1, "Waiting for OK sign from process " << pid
                 << ", child of thread " << ThisThreadId("") << endl);
      py2m_tsp->receive(&message, sizeof(message), recvd_size, priority);
      if (message == 0) {
        VERBOSE(1, "Received OK from process " << pid << endl);
        // And continue execution
      } else {
        VERBOSE(1, "PyMoses sent bad message!" << endl);
        Cleanup();
        UTIL_THROW(util::Exception, "Terminating!");
      }
    } else if (pid == 0) {
      // CHILD PROCESS; Start pymoses and pipe the results back
      if (!(fpipe = (FILE*)popen(command.c_str(), "r"))) {
        UTIL_THROW(util::Exception, "Problems with pymoses pipe command");
      }
      while (fgets( line, sizeof line, fpipe)) {
        // printf("%s", line);
      }
      pclose(fpipe);
      // We have to exit here, else it will try translating everything again
      // but die (because the shared memory/MQs have been deleted)
      exit(0);
    } else {
      Cleanup();
      UTIL_THROW(util::Exception, "Forking error");
    }
  }

  void CSLM::IssuePythonRequest(std::vector<const Word*> contextFactor) {
    // Increase the batch size
    int* n = batch_count.get();
    *n = *n + 1;

    // Create the n-gram from factor IDs
    // Note that these NumPy functions can be called without GIL
    NpyIter* iter = ngrams.get();
    NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
      NpyIter_Deallocate(iter);
      Cleanup();
      UTIL_THROW(util::Exception, "Unable to create Numpy iterator function");
    }
    int **dataptr = (int**) NpyIter_GetDataPtrArray(iter);
    for (unsigned int i = 0; i < contextFactor.size(); i++) {
      if(contextFactor[i]->GetFactor(1)) {
        **dataptr = contextFactor[i]->GetFactor(1)->GetIndex();
      } else {
        // TODO: DO not hardcode UNK index
        **dataptr = 1;
      }
      iternext(iter);
    }
  }

  void CSLM::Evaluate(const InputType &input,
                      const InputPath &inputPath,
                      const TargetPhrase &targetPhrase,
                      const StackVec *stackVec,
                      ScoreComponentCollection &scoreBreakdown,
                      ScoreComponentCollection *estimatedFutureScore) const {
    // This gets called for each target phrase before the search begins
    // We see if the source sentence has already been set
    // If not, we binarize it and store it in a NumPy array
    if (conditional) {
      bool reset = false;
      if (!source_sentence.get()) {
        reset = true;
      } else if (source_sentence->GetTranslationId() != input.GetTranslationId()) {
        reset = true;
      }
      if (reset) {
        source_length.reset(new int(input.GetSize()));
        source_sentence.reset(const_cast<InputType*>(&input));

        NpyIter* iter = source.get();
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(
          iter, (char**)"Unable to get source InterNextFunc"
        );
        if (!iternext) {
          const_cast<CSLM*>(this)->Cleanup();
          UTIL_THROW(util::Exception, "Unable to get source InterNextFunc");
        }
        int result_source = NpyIter_Reset(iter, (char**)"Unable to reset source");
        if (result_source == NPY_FAIL) {
          const_cast<CSLM*>(this)->Cleanup();
          UTIL_THROW(util::Exception, "Unable to reset source");
        }
        int **dataptr = (int**) NpyIter_GetDataPtrArray(iter);
        for (unsigned int i = 0; i < input.GetSize(); i++) {
          if(input.GetWord(i).GetFactor(1)) {
            **dataptr = input.GetWord(i).GetFactor(1)->GetIndex();
          } else {
            **dataptr = 1;
          }
          iternext(iter);
        }
      }
    }
  }

  void CSLM::Evaluate(const Phrase &source,
                      const TargetPhrase &targetPhrase,
                      ScoreComponentCollection &scoreBreakdown,
                      ScoreComponentCollection &estimatedFutureScore) const {
    // This gets called by the phrase table; However, we can't score
    // n-grams of variable length in isolation, so we don't do anything
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
    // TODO: This logic doesn't make sense! Non phrase-boundary n-grams
    // also need to be scored... This problem needs to be addressed 
    // here as well as in Implementation.cpp's Evaluate function
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
    }
  }

  LMResult CSLM::GetValue(const std::vector<const Word*> &contextFactor,
                          State* finalState) const {
    // Get the iterator of the scores
    NpyIter* iter = scores.get();
    NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, (char**)"Unable to get scores InterNextFunc");
    if (!iternext) {
      const_cast<CSLM*>(this)->Cleanup();
      UTIL_THROW(util::Exception, "Unable to get scores InterNextFunc");
    }
    float **dataptr = (float**) NpyIter_GetDataPtrArray(iter);

    // Construct the score result
    LMResult ret;
    ret.score = **dataptr;
    ret.unknown = false;

    // Increment iterator for next score
    iternext(iter);

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
    if (*batch_count.get() > 0) {
      int message = *batch_count.get();
      m2py_tsp->send(&message, sizeof(int), 0);
      if (conditional) {
        message = *source_length.get();
        m2py_tsp->send(&message, sizeof(int), 0);
      }
    }
  }

  void CSLM::SyncBuffer() {
    // Here we wait for the child process to finish scoring before we read out
    // the scores from shared memory
    if (*batch_count.get() > 0) {
      int message = 0;
      message_queue::size_type recvd_size;
      unsigned int priority;
      py2m_tsp->receive(&message, sizeof(message), recvd_size, priority);
      if (message != 1) {
        Cleanup();
        UTIL_THROW(util::Exception, "Received wrong message from PyMoses while waiting for eval");
      }
    }
  }

  void CSLM::ClearBuffer() {
    // All the hypotheses in this batch have been scored, so delete them from
    // the shared memory. NpyIter_Reset is called with a custom error message
    // so that it does not require the GIL
    int result_ngrams = NpyIter_Reset(ngrams.get(), (char**)"Unable to reset ngrams");
    if (result_ngrams == NPY_FAIL) {
      Cleanup();
      UTIL_THROW(util::Exception, "Unable to reset ngrams");
    }
    int result_scores = NpyIter_Reset(scores.get(), (char**)"Unable to reset scores");
    if (result_scores == NPY_FAIL) {
      Cleanup();
      UTIL_THROW(util::Exception, "Unable to reset scores");
    }
    int* n = batch_count.get();
    *n = 0;
  }

}

