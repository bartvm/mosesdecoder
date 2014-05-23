#include "CSLMPenalty.h"
#include "moses/TargetPhrase.h"
#include "moses/Factor.h"
#include "moses/ScoreComponentCollection.h"
#include "util/exception.hh"
#include <vector>

using namespace std;

namespace Moses
{

CSLMPenalty::CSLMPenalty(const std::string &line) : StatelessFeatureFunction(1, line), m_UNK(-1) {
  ReadParameters();
  UTIL_THROW_IF2(m_UNK == -1, "CSLMPenalty must receive factor argument");
}

void CSLMPenalty::Evaluate(const Phrase &source
                                   , const TargetPhrase &targetPhrase
                                   , ScoreComponentCollection &scoreBreakdown
                                   , ScoreComponentCollection &estimatedFutureScore) const
{
  float score = 0.0;
  for (unsigned int i = 0; i < targetPhrase.GetSize(); i++) {
    if (targetPhrase.GetFactor(i, m_factorType)->GetIndex(m_UNK) == m_UNK) {
      score += 1.0;
    }
  }
  vector<float> newScores(m_numScoreComponents);
  newScores[0] = score;
  scoreBreakdown.Assign(this, newScores);
}

void CSLMPenalty::Evaluate(const InputType &input
                                   , const InputPath &inputPath
                                   , const TargetPhrase &targetPhrase
                                   , const StackVec *stackVec
                                   , ScoreComponentCollection &scoreBreakdown
                                   , ScoreComponentCollection *estimatedFutureScore) const
{}

void CSLMPenalty::Evaluate(const Hypothesis& hypo,
                                   ScoreComponentCollection* accumulator) const
{}

void CSLMPenalty::EvaluateChart(const ChartHypothesis &hypo,
                                        ScoreComponentCollection* accumulator) const
{}

void CSLMPenalty::SetParameter(const std::string& key, const std::string& value) {
  if (key == "UNK") {
    m_UNK = Scan<int>(value);
  } else if (key == "factor") {
    m_factorType = Scan<FactorType>(value);
  }
}

}

