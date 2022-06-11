/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file auto_scheduler/search_policy/search_policy.cc
 * \brief The base class of search policies.
 */

#include <tvm/auto_scheduler/measure_record.h>
#include <tvm/auto_scheduler/search_policy.h>
#include <tvm/runtime/registry.h>

#include "utils.h"

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_OBJECT_TYPE(SearchCallbackNode);
TVM_REGISTER_OBJECT_TYPE(SearchPolicyNode);
TVM_REGISTER_OBJECT_TYPE(PreloadMeasuredStatesNode);

void SearchPolicyNode::PreloadMeasuredStates(const String& log_file) {
  RecordReader reader = RecordReader(log_file);
  const auto& res = reader->ReadLines(-1);
  size_t log_size = res.first.size();
  ICHECK_EQ(log_size, res.second.size());
  if (log_size) {
    Array<State> measured_states;
    std::vector<float> measured_throughputs;
    for (size_t i = 0; i < log_size; i++) {
      const auto& inp = res.first[i];
      if (inp->task->workload_key == search_task->workload_key &&
          inp->task->target->kind->name.compare(search_task->target->kind->name) == 0) {
        State state = search_task->compute_dag->init_state;
        auto pstate = state.CopyOnWrite();
        pstate->transform_steps = inp->state->transform_steps;
        for (const auto& step : pstate->transform_steps) {
          StepApplyToState(step, &state, search_task->compute_dag);
        }
        measured_states.push_back(std::move(state));
        measured_throughputs.push_back(
            res.second[i]->error_no == 0 ? (1.0 / FloatArrayMean(res.second[i]->costs)) : 0.0);
      }
    }
    // We can assume the recorded states will all be valid after infer bound
    measured_states = search_task->compute_dag.InferBound(measured_states);
    for (size_t i = 0; i < measured_states.size(); i++) {
      auto& state = measured_states[i];
      const auto& state_str = state.ToStr();
      if (!measured_states_set_.count(state_str)) {
        measured_states_set_.insert(state_str);
        if (measured_throughputs[i] != 0.0) {
          measured_states_vector_.emplace_back(std::move(state));
          measured_states_throughputs_.emplace_back(measured_throughputs[i]);
        }
      }
    }

    StdCout(verbose) << "SearchPolicy: Loaded " << measured_states_set_.size()
                     << " measurement records from " << log_file << " for "
                     << search_task->workload_key << std::endl;
  } else {
    StdCout(verbose) << "SearchPolicy: No measurement records found in " << log_file << " for "
                     << search_task->workload_key << std::endl;
  }
}

void SearchPolicyNode::RunCallbacks(const Array<SearchCallback>& callbacks) {
  for (const auto& callback : callbacks) {
    callback->Callback(this);
  }
}

std::string SearchPolicyNode::ExtractSystemCmdOutput(const char* cmd) {
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  return result;
}

float SearchPolicyNode::ComputeStdFromVector(const std::vector<float>& data) {
  float sum = std::accumulate(data.begin(), data.end(), 0.0f);
  float mean = sum / data.size();
  float accum = 0.0f;
  std::for_each(data.begin(), data.end(), [&](const float x) {
    accum += (x - mean) * (x - mean);
  });
  return std::sqrt(accum / (data.size() - 1));
}

std::vector<float> SearchPolicyNode::ExtractProfileResult(
    const std::string& parse_script,
    const std::string& prof_file) {
  std::string cmd = "python3 " + parse_script + " --log-file=" + prof_file;
  std::string output = ExtractSystemCmdOutput(cmd.c_str());
  std::vector<std::string> values = SplitStrByNewLine(output);
  std::vector<float> float_values;
  for (const std::string& value : values) {
    float_values.push_back(std::stof(value));
  }
  return float_values;
}

int SearchPolicyNode::GetLogLineNum(const char* log_file) {
  // NOTE: here we assume there is no new line at the end of file, and the log
  // file produced by Ansor ensures this.
  std::ifstream fs(log_file);
  int line_num = std::count(std::istreambuf_iterator<char>(fs),
                            std::istreambuf_iterator<char>(), '\n');
  return line_num;
}

std::vector<Array<MeasureResult>> SearchPolicyNode::Profile(
    const SearchTask& task, const Array<MeasureInput>& inputs, int batch_size) {
  std::vector<Array<MeasureResult>> prof_results;
  std::vector<std::vector<float>> metric_values;

  std::string dir = "/home/shiyuw3/Research/Profiling-Guided-Tuning-Sketch/"
                    "experiments/ansor/single_op/";
  std::string log_file = dir + "tmp.log";
  std::string prof_file = dir + "tmp-ncu.log";
  std::string exec_script = dir + "tune_single_op.py";
  std::string parse_script = dir + "parse_profile.py";
  int num_record = GetLogLineNum(log_file.c_str());
  int num_input = inputs.size();

  for (int i = 0; i < num_input; ++i) {
    int idx = num_record - num_input + i;
    RunProfiler(exec_script, log_file, prof_file, idx);
    std::vector<float> values = ExtractProfileResult(parse_script, prof_file);
    for (float value : values) {
      StdCout(verbose) << std::to_string(value) << "\n";
    }
    metric_values.push_back(values);
  }

  for (size_t i = 0; i < metric_values[0].size(); ++i) {
    Array<MeasureResult> results;
    for (size_t j = 0; j < metric_values.size(); ++j) {
      std::vector<float> values = metric_values[j];
      Array<PrimExpr> costs;
      costs.push_back(values[i]);
      MeasureResult result = MeasureResult(costs,
                                           /* error_no= */0,
                                           /* error_msg= */"",
                                           /* all_cost= */0.0,
                                           /* timestamp= */0.0);
      results.push_back(result);
    }
    prof_results.push_back(results);
  }

  return prof_results;
}

void SearchPolicyNode::RunProfiler(const std::string& exec_script,
                                   const std::string& log_file,
                                   const std::string& prof_file,
                                   int idx) {
  std::string cmd = "ncu --set full --csv --details-all -c 10 python3 ";
  std::string workload = "sample-conv2d";
  cmd += exec_script + " --wkl " + workload + " --eval-trial-index " +
         std::to_string(idx) + " --log-file " + log_file + " > " + prof_file;
  StdCout(verbose) << cmd << "\n";
  system(cmd.c_str());
}

std::vector<std::string> SearchPolicyNode::SplitStrByNewLine(
    const std::string& str) {
  std::vector<std::string> tokens;

  std::stringstream ss(str);
  std::string token;
  while (std::getline(ss, token, '\n')) {
    tokens.push_back(token);
  }

  return tokens;
}

PreloadMeasuredStates::PreloadMeasuredStates(String filename) {
  auto node = make_object<PreloadMeasuredStatesNode>();
  node->filename = std::move(filename);
  data_ = std::move(node);
}

void PreloadMeasuredStatesNode::Callback(SearchPolicyNode* policy) {
  policy->PreloadMeasuredStates(filename);
}

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyRunCallbacks")
    .set_body_typed([](SearchPolicy policy, Optional<Array<SearchCallback>> callbacks) {
      if (callbacks) {
        policy->RunCallbacks(callbacks.value());
      }
    });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyContinueSearchOneRound")
    .set_body_typed([](SearchPolicy policy, int num_measure, ProgramMeasurer measurer) {
      Array<MeasureInput> inputs;
      Array<MeasureResult> results;
      std::tie(inputs, results) = policy->ContinueSearchOneRound(num_measure, measurer);
      return Array<ObjectRef>{inputs, results};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicySetVerbose")
    .set_body_typed([](SearchPolicy policy, int verbose) { policy->verbose = verbose; });

TVM_REGISTER_GLOBAL("auto_scheduler.PreloadMeasuredStates").set_body_typed([](String filename) {
  return PreloadMeasuredStates(filename);
});

}  // namespace auto_scheduler
}  // namespace tvm
