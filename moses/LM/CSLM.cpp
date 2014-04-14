
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

  // needed by parent language model classes. Why didn't they set these themselves?
  m_sentenceStart = factorCollection.AddFactor(Output, m_factorType, BOS_);
  m_sentenceStartWord[m_factorType] = m_sentenceStart;

  m_sentenceEnd		= factorCollection.AddFactor(Output, m_factorType, EOS_);
  m_sentenceEndWord[m_factorType] = m_sentenceEnd;
}

CSLM::~CSLM()
{
}

void CSLM::Load()
{
  VERBOSE(1, "Starting Python" << std::endl);
  Py_Initialize();
  try {
    boost::python::object py_cslm = boost::python::import("cslm_pool");
    py_run_cslm = py_cslm.attr("run_cslm");
  } catch(boost::python::error_already_set const &) {
    PyErr_Print();
    UTIL_THROW2("Unable to load Python module and/or function");
  }
}

void CSLM::SendBuffer()
{
  boost::lock_guard<boost::mutex> guard(py_mtx);
  py_run_cslm();
}

void CSLM::ClearBuffer()
{

}

void CSLM::SyncBuffer()
{

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



