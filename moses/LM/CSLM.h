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
    std::vector<std::vector<const Word*> > requests;
    float cslm_score;
    int requests_count = 0;
    mutable int value_count = 0;
    object py_run_cslm;
    object py_request;
    void IssuePythonRequest(std::vector<const Word*> contextFactor);
    
    
  public:
    CSLM(const std::string &line);
    void Load();
    ~CSLM();
    
    virtual LMResult GetValue(const std::vector<const Word*> &contextFactor, State* finalState = 0) const;
    
    void SetFFStateIdx(int state_idx);
    void sync();
    
    virtual void IssueRequestsFor(Hypothesis& hypo, const FFState* input_state);
    
  };
  
  
}
