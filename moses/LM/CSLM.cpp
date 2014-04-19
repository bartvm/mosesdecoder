#include "CSLM.h"
#include <sys/wait.h>

using namespace std;
using namespace boost::interprocess;

namespace Moses {
  CSLM::CSLM(const std::string &line)
    : LanguageModelSingleFactor(line),
      ngrams(&NpyIterCleanup), scores(&NpyIterCleanup) {
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
    import_array();
    // Save the current thread state, release implicit GIL
    state = PyEval_SaveThread();
  }

  string CSLM::ThisThreadId(string suffix) const {
    // For reasons beyond my comprehension, storing the thread ID in a
    // variable doesn't work; multiple threads end up getting the same ID
    // and the messaging queues/shared memory crash. Requesting the thread ID
    // through a function like this seems to work (note: using a
    // thread_shared_ptr isn't any better and results in double free,
    // corruption, wrong pointers, and more!)
    std::stringstream ss;
    ss << boost::this_thread::get_id() << suffix;
    return ss.str();
  }

  void CSLM::StopThread() {
    // Delete the message queues as soon as PyMoses responds
    int message = -1;
    m2py_tsp->send(&message, sizeof(int), 0);
    message_queue::size_type recvd_size;
    unsigned int priority;
    py2m_tsp->receive(&message, sizeof(message), recvd_size, priority);
    message_queue::remove(ThisThreadId("m2py").c_str());
    message_queue::remove(ThisThreadId("py2m").c_str());

    // We wait for our Moses-fork to exit
    waitpid(*child_pid.get(), NULL, 0);

    // Clean up shared memory
    shared_memory_object::remove(
      ThisThreadId("ngrams").c_str()
    );
    shared_memory_object::remove(
      ThisThreadId("scores").c_str()
    );
  }

  void CSLM::LoadThread() {
    // Setting up message queues
    VERBOSE(1, "Setting up message queues" << endl);
    // 0 signals READY
    // 1 signals NEXT
    // 2 signals EXIT
    // other ERROR

    m2py_tsp.reset(new message_queue(open_or_create,
                                     ThisThreadId("m2py").c_str(), 1, sizeof(int)));
    py2m_tsp.reset(new message_queue(open_or_create,
                                     ThisThreadId("py2m").c_str(), 1, sizeof(int)));

    // Setting up the shared memory segment
    // Easier doing it this way than in a separate function
    // because some things seem to disappear in the stack
    VERBOSE(1, "Setting up shared memory" << endl);
    shared_memory_object ngrams_shm_obj(open_or_create, ThisThreadId("ngrams").c_str(), read_write);
    shared_memory_object scores_shm_obj(open_or_create, ThisThreadId("scores").c_str(), read_write);
    // TODO: These numbers should be calculated more precisely (now 1MB for each)
    ngrams_shm_obj.truncate(1048576);
    scores_shm_obj.truncate(1048576);
    ngrams_region_tsp.reset(new mapped_region(ngrams_shm_obj, read_write));
    scores_region_tsp.reset(new mapped_region(scores_shm_obj, read_write));
    memset(ngrams_region_tsp->get_address(), 1, ngrams_region_tsp->get_size());
    memset(scores_region_tsp->get_address(), 1, scores_region_tsp->get_size());

    // Create the Python NumPy wrappers and store iterators over them
    // We need the GIL for this!
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    int ngrams_nd = 2;
    npy_intp ngrams_dims[2] = {10000, 7};
    PyObject* ngrams_array = PyArray_SimpleNewFromData(ngrams_nd, ngrams_dims,
                                                       NPY_INT,
                                                       ngrams_region_tsp->get_address());
    NpyIter *ngrams_iter = NpyIter_New((PyArrayObject*)ngrams_array,
                                       NPY_ITER_READWRITE, NPY_KEEPORDER,
                                       NPY_NO_CASTING, NULL);
    Py_DECREF(ngrams_array);
    ngrams.reset(ngrams_iter);

    int scores_nd = 1;
    npy_intp scores_dims[1] = {10000};
    PyObject* scores_array = PyArray_SimpleNewFromData(scores_nd, scores_dims,
                                                       NPY_FLOAT,
                                                       scores_region_tsp->get_address());
    NpyIter *scores_iter = NpyIter_New((PyArrayObject*)scores_array,
                                       NPY_ITER_READWRITE, NPY_KEEPORDER,
                                       NPY_NO_CASTING, NULL);
    Py_DECREF(scores_array);
    scores.reset(scores_iter);

    PyGILState_Release(gstate);

    batch_count.reset(new int(0));

    // Create the PyMoses command to execute and pipe PyMoses's stdout back
    // to the parent
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
      child_pid.reset(new int(pid));
      int message = 1;
      message_queue::size_type recvd_size;
      unsigned int priority;
      VERBOSE(1, "Waiting for OK sign from process " << pid
                 << ", child of thread " << ThisThreadId("") << endl);
      py2m_tsp->receive(&message, sizeof(message), recvd_size, priority);
      if (message == 0) {
        VERBOSE(1, "Received OK from process " << pid << endl);
        // And continue execution
      } else {
        VERBOSE(1, "PyMoses sent bad message!" << endl);
        // TODO: Clean exit
      }
    } else if (pid == 0) {
      // CHILD PROCESS; Start pymoses and pipe the results back
      if (!(fpipe = (FILE*)popen(command.c_str(), "r"))) {
        VERBOSE(1, "Problems with pipe" << endl);
        // TODO: Clean exit
      }
      while (fgets( line, sizeof line, fpipe)) {
        // printf("%s", line);
      }
      pclose(fpipe);
      // We have to exit here, else it will try translating everything again
      // but die (because the shared memory/MQs have been deleted)
      exit(0);
    } else {
      VERBOSE(1, "Forking error" << endl);
      // TODO: Clean exit
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
      VERBOSE(1, "Unable to create Numpy iterator function" << endl);
      // TODO: Clean exit
    }
    int **dataptr = (int**) NpyIter_GetDataPtrArray(iter);
    for (unsigned int i = 0; i < contextFactor.size(); i++) {
      if(contextFactor[i]->GetFactor(1)) {
        try {
          int cslm_id = boost::lexical_cast<int>(
            contextFactor[i]->GetString(1).as_string()
          );
          **dataptr = cslm_id;
        } catch (const boost::bad_lexical_cast& e) {
          **dataptr = 1;
          VERBOSE(3, "Python error: Got non-integer CSLM ID, defaulted to 1 ('"
                      << contextFactor[i]->GetString(1).as_string()
                      << "' for word '"
                      << contextFactor[i]->GetString(0).as_string() << "')"
                      << endl);
        }
      } else {
        **dataptr = 1;
      }
      iternext(iter);
    }
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
    // Get the iterator of the scores
    NpyIter* iter = scores.get();
    NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
    int **dataptr = (int**) NpyIter_GetDataPtrArray(iter);

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
        VERBOSE(1, "Received wrong message from PyMoses while waiting for eval"
                   << endl);
        // TODO: Clean exit
      }
    }
  }

  void CSLM::ClearBuffer() {
    // All the hypotheses in this batch have been scored, so delete them from
    // the shared memory
    NpyIter_Reset(ngrams.get(), NULL);
    NpyIter_Reset(scores.get(), NULL);
    int* n = batch_count.get();
    *n = 0;
  }

}

