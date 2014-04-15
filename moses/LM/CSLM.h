// $Id$
#pragma once

#include <vector>
#include "SingleFactor.h"

#include <boost/thread.hpp>
#include "Python.h"
#include <numpy/ndarrayobject.h>
#include "util/exception.hh"
#include "moses/Util.h"
#include "moses/StaticData.h"


namespace Moses
{

class CSLM : public LanguageModelSingleFactor
{
protected:
  // Import Python modules and methods
  PyObject *pFunc, *pModule;
  // This mutex prevents simultaneous calls to Python
  boost::mutex mtx_;

  // Thread local storage of batches and results
  boost::thread_specific_ptr<bool> loaded;
  boost::thread_specific_ptr<PyObject> batch(&Py_XDECREF);
  boost::thread_specific_ptr<PyObject> scores(&Py_XDECREF);
  boost::thread_specific_ptr<PyObject> async_result(&Py_XDECREF);

  // Add factor IDs for EOS and BOS markers
  const Factor *m_sentenceStart_CSLM, *m_sentenceEnd_CSLM;

  void IssueRequestsFor(Hypothesis& hypo, const FFState* input_state);
  void IssueRequestFor(std::vector<const Word*>);


public:
  CSLM(const std::string &line);
  ~CSLM();
  void Load();

  void SendBuffer();
  void SyncBuffer();
  void ClearBuffer();

  virtual LMResult GetValue(const std::vector<const Word*> &contextFactor, State* finalState = 0) const;
};


}
