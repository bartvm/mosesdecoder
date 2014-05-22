#include "CSLMPenalty.h"
#include "moses/TargetPhrase.h"
#include "moses/Factor.h"
#include "moses/ScoreComponentCollection.h"
#include <vector>

using namespace std;

namespace Moses
{

void CSLMPenalty::Evaluate(const Phrase &source
                                   , const TargetPhrase &targetPhrase
                                   , ScoreComponentCollection &scoreBreakdown
                                   , ScoreComponentCollection &estimatedFutureScore) const
{
  float score = 0.0;
  for (unsigned int i = 0; i < targetPhrase.GetSize(); i++) {
    if (targetPhrase.GetFactor(i, 1)->GetIndex() == 1) {
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

}

