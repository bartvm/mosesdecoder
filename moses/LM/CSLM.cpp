
#include "CSLM.h"
#include "moses/FactorCollection.h"
#include "Python.h"

using namespace std;

namespace Moses
{
CSLM::CSLM(const std::string &line)
  :LanguageModelSingleFactor(line)
{
  ReadParameters();

  FactorCollection &factorCollection = FactorCollection::Instance();

  // needed by parent language model classes. Why didn't they set these themselves?
  m_sentenceStart = factorCollection.AddFactor(Output, m_factorType, BOS_);
  m_sentenceStartWord[m_factorType] = m_sentenceStart;

  m_sentenceEnd		= factorCollection.AddFactor(Output, m_factorType, EOS_);
  m_sentenceEndWord[m_factorType] = m_sentenceEnd;
}

CSLM::~CSLM()
{
  Py_DECREF(pFunc);
  Py_Finalize();
}

void CSLM::Load()
{
  TRACE_ERR("Loading CSLM");
  Py_Initialize();
  std::string moduleName = "lm";
  std::string functionName = "get_score";
  PyObject *pName = PyString_FromString(moduleName.c_str());
  PyObject *pModule = PyImport_Import(pName);
  Py_DECREF(pName);
  if (pModule != NULL) {
    pFunc = PyObject_GetAttrString(pModule, functionName.c_str());
    if (pFunc && PyCallable_Check(pFunc)) {
      PyObject *pArg = PyTuple_New(1);
      PyTuple_SetItem(pArg, 0, PyString_FromString("this is a test string"));
      PyObject *pValue = PyObject_CallObject(pFunc, pArg);
      Py_DECREF(pArg);
      if (pValue != NULL) {
        Py_INCREF(pFunc);
        cout << "Python module succesfully loaded and tested" << endl;
      } else {
        cout << "Call to Python function failed" << endl;
      }
    } else {
      cout << "Cannot find Python function" << endl;
    }
  } else {
    cout << "Failed to import Python module" << endl;
  }
}

LMResult CSLM::GetValue(const vector<const Word*> &contextFactor, State* finalState) const
{
  std::string phrase = "";
  for (size_t i=0, n=contextFactor.size(); i<n; i+=1) {
    const Word* word = contextFactor[i];
    const Factor* factor = word->GetFactor(m_factorType);
    const std::string string = factor->GetString().as_string();
    phrase = phrase + " " + string;
  }

  double value;
  PyObject *pArg = PyTuple_New(1);
  PyTuple_SetItem(pArg, 0, PyString_FromString(phrase.c_str()));
  if (pFunc && PyCallable_Check(pFunc)) {
    PyObject *pValue = PyObject_CallObject(pFunc, pArg);
    Py_DECREF(pArg);
    if (pValue != NULL) {
      value = PyFloat_AsDouble(pValue);
    } else {
      cout << "Call to Python function failed, 0 score" << endl;
      value = 0.0;
    }
  } else {
    cout << "Cannot access Python function, 0 score" << endl;
    value = 0.0;
  }

  LMResult ret;
  ret.score = value;
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



