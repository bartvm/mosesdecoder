// $Id$
#pragma once

#include <vector>
#include "SingleFactor.h"
#include "Python.h"

namespace Moses
{

class CSLM : public LanguageModelSingleFactor
{
protected:
  PyObject *pFunc;
public:
  CSLM(const std::string &line);
  ~CSLM();

  virtual LMResult GetValue(const std::vector<const Word*> &contextFactor, State* finalState = 0) const;
  virtual void Load();
};


}
