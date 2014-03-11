
#include "CSLM.h"
#include "moses/FactorCollection.h"
#include "PointerState.h"
#include <iostream>
#include <boost/python/suite/indexing/map_indexing_suite.hpp> // Needed?
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

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
    class_<map<vector<string>,float> >("ContextFactors")
    .def(map_indexing_suite<map<vector<string>, float> >());
    class_<vector<string> >("Phrase")
    .def(vector_indexing_suite<vector<string> >());
//    class_<Factor, Factor*, boost::noncopyable>("Factor", no_init)
//    .def("GetString", &Factor::GetString);
//    class_<Word, boost::remove_const<Word*>::type >("Word")
//    .def("GetFactor", &Word::GetFactor, return_value_policy<reference_existing_object>());
    try {
      object py_cslm = import("cslm");
//      py_request = py_cslm.attr("request");
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
    vector<string> phrase;
    for (int i = 0; i < contextFactor.size(); i++) {
      phrase.push_back(contextFactor[i]->GetString(0).as_string());
    }
    requests.insert(pair<vector<string>, float>(phrase, 0.0));
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
  
  LMResult CSLM::GetValue(const vector<const Word*> &contextFactor,
                          State* finalState) const {
    LMResult ret;
//    pair<map<vector<const Word*>, float>::iterator, bool> map_lookup
    vector<string> phrase;
    for (int i = 0; i < contextFactor.size(); i++) {
      phrase.push_back(contextFactor[i]->GetString(0).as_string());
    }
    map<vector<string>, float>::const_iterator map_lookup = requests.find(phrase);
    if (map_lookup == requests.end()) {
      // Throw an error!
      cout << "ERROR" << endl;
    } else {
      ret.score = map_lookup->second;
    }
//    ret.score = requests.contextFactor];
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
  
  void CSLM::SyncBuffer() {
    if (requests.size() > 0) {
      try {
        object scores = py_run_cslm(requests);
      } catch(error_already_set const &) {
        PyErr_Print();
      }
    }
  }
  
  void CSLM::ClearBuffer() {
    requests.clear();
  }
  
}



