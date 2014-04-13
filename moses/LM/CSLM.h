// $Id$
#pragma once

#include <vector>
#include "SingleFactor.h"

#include <boost/thread.hpp>
#include <boost/python.hpp>
#include "util/exception.hh"
#include "moses/Util.h"

namespace Moses
{

class CSLM : public LanguageModelSingleFactor
{
protected:
  boost::python::object py_run_cslm;
  boost::mutex py_mtx;

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
