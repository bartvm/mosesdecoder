
#include "CSLM.h"
#include "moses/FactorCollection.h"

using namespace std;

namespace Moses
{
CSLM::CSLM(const std::string &line)
  :LanguageModelSingleFactor(line)
{
  ReadParameters();

  FactorCollection &factorCollection = FactorCollection::Instance();

  // Needed by parent language model classes. Why didn't they set these themselves?
  // Here we also set the BOS and EOS markers to 0
  m_sentenceStart = factorCollection.AddFactor(Output, m_factorType, BOS_);
  m_sentenceStart_CSLM = factorCollection.AddFactor(Output, 1, "0");
  m_sentenceStartWord[m_factorType] = m_sentenceStart;
  m_sentenceStartWord[1] = m_sentenceStart_CSLM;

  m_sentenceEnd = factorCollection.AddFactor(Output, m_factorType, EOS_);
  m_sentenceEnd_CSLM = factorCollection.AddFactor(Output, 1, "0");
  m_sentenceEndWord[m_factorType] = m_sentenceEnd;
  m_sentenceEndWord[1] = m_sentenceEnd_CSLM;
}

CSLM::~CSLM()
{
  // This should deallocate all memory
  Py_Finalize();
}

void CSLM::Load()
{
  // Threads have not been loaded yet
  loaded.reset(new bool(false));
  VERBOSE(1, "Starting Python" << std::endl);
  Py_Initialize();
  import_array();
  PyObject* pName = PyString_FromString("cslm_pool");
  pModule = PyImport_Import(pName);
  Py_DECREF(pName);
  if (pModule != NULL) {
    pFunc = PyObject_GetAttrString(pModule, "apply_async");
    if (!pFunc || !PyCallable_Check(pFunc)) {
      if (PyErr_Occurred()) {
        PyErr_Print();
      }
      UTIL_THROW2("Unable to load Python method apply_async");
    } else {
      VERBOSE(1, "Successfully imported" << std::endl);
    }
  } else {
    if (PyErr_Occurred()) {
      PyErr_Print();
    }
    UTIL_THROW2("Unable to load Python module cslm_pool");
  }
}

void CSLM::IssueRequestsFor(Hypothesis& hypo, const FFState* input_state)
{
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
  IssueRequestFor(contextFactor);

  // Main loop
  size_t endPos = std::min(startPos + GetNGramOrder() - 2, currEndPos);
  for (size_t currPos = startPos + 1 ; currPos <= endPos ; currPos++) {
    // Shift all args down 1 place
    for (size_t i = 0 ; i < GetNGramOrder() - 1 ; i++) {
      contextFactor[i] = contextFactor[i + 1];
    }
    // Add last factor
    contextFactor.back() = &hypo.GetWord(currPos);
    IssueRequestFor(contextFactor);
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
    IssueRequestFor(contextFactor);
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

void CSLM::IssueRequestFor(std::vector<const Word*> contextFactor)
{
  // This gets called only once for each thread
  if (!loaded.get()) {
    loaded.reset(new bool(true));
    // Construct the NumPy array in which to store ngrams
    npy_intp batch_dims[2] = {1000, 7};
    npy_intp scores_dims[1] = {1000};
    PyObject *pBatch = PyArray_ZEROS(2, batch_dims, NPY_INT, 0);
    PyObject *pScores = PyArray_ZEROS(1, scores_dims, NPY_FLOAT, 0);
    batch.reset(pBatch);
    scores.reset(pScores);
    // INCREF to prevent cleanup by Python GC?
    Py_INCREF(pBatch);
    Py_INCREF(pScores);
    VERBOSE(3, "Buffer set" << std::endl);
  }
  // Create the n-gram from factor IDs
  // for (unsigned int i = 0; i < contextFactor.size(); i++) {
  //   if(contextFactor[i]->GetFactor(1)) {
  //     try {
  //       int cslm_id = boost::lexical_cast<int>(
  //         contextFactor[i]->GetString(1).as_string()
  //       );
  //       phrase.push_back(cslm_id);
  //     } catch (const boost::bad_lexical_cast& e) {
  //       phrase.push_back(1);
  //       VERBOSE(3, "Python error: Got non-integer CSLM ID, defaulted to 1 ('"
  //                   << contextFactor[i]->GetString(1).as_string()
  //                   << "' for word '"
  //                   << contextFactor[i]->GetString(0).as_string() << "')"
  //                   << endl);
  //     }
  //   } else {
  //     phrase.push_back(1);
  //   }
  // }
  // MapType *requests = segment->find<MapType>("MyMap").first;
  // Insert the n-gram with a placeholder score of 0.0
  // requests->insert(MapElementType(phrase, 0.0));
  // score_map->insert(std::make_pair(contextFactor, phrase));
  // batch.reset(new int(10));
}

void CSLM::SendBuffer()
{
  VERBOSE(1, "Calling function" << std::endl);
  mtx_.lock();
  PyObject *pArgs = PyTuple_New(1);
  PyTuple_SetItem(pArgs, 0, batch.get());
  PyObject *pAsyncResult = PyObject_CallObject(pFunc, pArgs);
  // async_result.reset(pAsyncResult);
  // Py_INCREF(pAsyncResult);
  mtx_.unlock();
}

void CSLM::ClearBuffer()
{

}

void CSLM::SyncBuffer()
{
  VERBOSE(1, "Waiting for results" << std::endl);
}

LMResult CSLM::GetValue(const vector<const Word*> &contextFactor, State* finalState) const
{
  LMResult ret;
  ret.score = contextFactor.size();
  ret.unknown = false;

  // use last word as state info
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

}



