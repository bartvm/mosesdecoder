
#include "CSLM.h"
#include "moses/FactorCollection.h"
#include "PointerState.h"
#include <iostream>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp> // Needed?

using namespace std;
using namespace boost::python;

namespace Moses {
  CSLM::CSLM(const std::string &line) : LanguageModelSingleFactor(line) {
    ReadParameters();
    
    FactorCollection &factorCollection = FactorCollection::Instance();
    
    // Needed by parent language model classes
    m_sentenceStart = factorCollection.AddFactor(Output, m_factorType, BOS_);
    m_sentenceStartWord[m_factorType] = m_sentenceStart;
    
    m_sentenceEnd		= factorCollection.AddFactor(Output, m_factorType, EOS_);
    m_sentenceEndWord[m_factorType] = m_sentenceEnd;
  }
  
  CSLM::~CSLM() {}
  
  void CSLM::Load() {
    Py_Initialize();
    try {
      object py_cslm = import("cslm");
      py_request = py_cslm.attr("request");
      py_run_cslm = py_cslm.attr("run_cslm");
    } catch(error_already_set const &) {
      PyErr_Print();
    }
    // Note that Boost does not support Py_Finalize();
  }
  
  void CSLM::SetFFStateIdx(int state_idx) {
    m_state_idx = state_idx;
  }
  
  void CSLM::IssuePythonRequest(vector<const Word*> contextFactor) {
    // Send a phrase to the Python module for word-id lookup
    boost::python::list py_phrase;
    for (int i = 0; i < contextFactor.size(); i++) {
      py_phrase.append(contextFactor[i]->GetString(0).as_string());
    }
    try {
      py_request(py_phrase);
    } catch(error_already_set const &) {
      PyErr_Print();
    }
  }
  
  void CSLM::IssueRequestsFor(Hypothesis& hypo, const FFState* input_state) {
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
    vector<const Word*> contextFactor(GetNGramOrder());
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
      requests.push_back(contextFactor);
      requests_count++;
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
  
  LMResult CSLM::GetValue(const vector<const Word*> &contextFactor,
                          State* finalState) const {
    LMResult ret;
    ret.score = 0.0;
    ret.unknown = false;
    
    // Get scores from sync here
    
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
  
  void CSLM::sync() {
    try {
      object scores = py_run_cslm();
    } catch(error_already_set const &) {
      PyErr_Print();
    }
  }
  
}



