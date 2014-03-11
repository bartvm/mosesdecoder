// $Id$
#pragma once

#include <vector>
#include "SingleFactor.h"
#include "moses/Hypothesis.h"
#include <boost/python.hpp>

using namespace boost::python;

namespace Moses
{
  
  class CSLM : public LanguageModelSingleFactor
  {
  protected:
    int m_state_idx;
    object py_run_cslm;
    void IssuePythonRequest(std::vector<const Word*> contextFactor);
    std::map<std::vector<std::string>, float> requests;
    
  public:
    CSLM(const std::string &line);
    void Load();
    ~CSLM();
    
    virtual LMResult GetValue(const std::vector<const Word*> &contextFactor, State* finalState = 0) const;
    
    void SetFFStateIdx(int state_idx);
    void SyncBuffer();
    void ClearBuffer();
    
    virtual void IssueRequestsFor(Hypothesis& hypo, const FFState* input_state);
    
  };
  
  
}
