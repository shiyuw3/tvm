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

#include <tvm/tir/expr.h>
#include <tvm/arith/analyzer.h>
#include <tvm/auto_scheduler/feature.h>
#include <tvm/auto_scheduler/measure.h>
#include <tvm/auto_scheduler/measure_record.h>
#include <tvm/driver/driver_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/support/parallel_for.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/op_attr_types.h>

#include <tvm/auto_scheduler/lstm_extractor.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "search_policy/utils.h"
#include "utils.h"

namespace tvm {
namespace auto_scheduler {

using namespace tvm::tir;
using arith::Analyzer;

// for loop
void FeatureVisitor::VisitStmt_(const ForNode* op) {
  const auto* extent = op->extent.as<IntImmNode>();
  int64_t loop_extent = -1;
  if (extent != nullptr) loop_extent = extent->value;
  AnnotationType ann = kSerial;
  switch (op->kind) {
    case ForKind ::kParallel:
      ann = kParallel;
      break;
    case ForKind::kUnrolled:
      ann = kUnrolled;
      break;
    case ForKind::kVectorized:
      ann = kVectorized;
      break;
    case ForKind::kSerial:
      ann = kSerial;
      break;
    case ForKind::kThreadBinding:
      LOG(FATAL) << "Loop ThreadBinding is reserved for future used and "
                 << "not yet supported in TIR";
      break;
    default:
      LOG(FATAL) << "Unknown annotation type";
      break;
  }

  if (EnterItervar_(op->loop_var, loop_extent, ann)) {
    StmtExprVisitor::VisitStmt_(op);
    ExitItervar_();
  }
}

// parallel axis, virtual thread
void FeatureVisitor::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tir::attr::thread_extent ||
      op->attr_key == tir::attr::virtual_thread) {
    Var var = op->node.as<tir::IterVarNode>()->var;
    const auto* extent = op->value.as<IntImmNode>();
    ICHECK(extent);

    std::string name = var.get()->name_hint;
    AnnotationType ann = kParallel;
    if (op->attr_key == tir::attr::thread_extent) {
      if (name == "blockIdx.x")
        ann = kBlockX;
      else if (name == "blockIdx.y")
        ann = kBlockY;
      else if (name == "blockIdx.z")
        ann = kBlockZ;
      else if (name == "threadIdx.x")
        ann = kThreadX;
      else if (name == "threadIdx.y")
        ann = kThreadY;
      else if (name == "threadIdx.z")
        ann = kThreadZ;
      else
        LOG(FATAL) << "invalid thread itervar " + name;
    } else {
      ann = kVirtualThread;
    }

    if (EnterItervar_(var, extent->value, ann)) {
      StmtExprVisitor::VisitStmt_(op);
      ExitItervar_();
    }
  } else {
    StmtExprVisitor::VisitStmt_(op);
  }
}

// memory access
void FeatureVisitor::VisitExpr_(const LoadNode* op) {
  EnterMem_(op->buffer_var, op->index);
  StmtExprVisitor::VisitExpr_(op);
  ExitMem_();
}

void FeatureVisitor::VisitStmt_(const StoreNode* op) {
  EnterMem_(op->buffer_var, op->index);
  StmtExprVisitor::VisitStmt_(op);
  ExitMem_();
}

// extract iter vars and their touch pattern from ir
bool TouchExtractor::EnterItervar_(
    Var var, int64_t length, AnnotationType ann_type) {
  // do not insert duplicated occurrences of virtual thread
  if (ann_type == kVirtualThread && itervar_map.count(var) != 0) {
    skip_stack_size_.push_back(itervar_stack_.size());
  } else {
    itervar_stack_.push_back(var);
    topdown_product_ *= length;

    if (itervar_map.count(var) != 0) {
      // find two duplicated axes
      // these happens when we create tvm.thread_axis("threadIdx.x") once and
      // bind it twice. Here we treat them as two axes
      // so we create a snapshot for the old one and freeze it
      Var old = Var(var.get()->name_hint);
      itervar_map.insert({old, itervar_map[var]});
      itervar_map.erase(var);
    }

    itervar_map.insert(
        {var, ItervarFeature(var, length,
                             static_cast<int>(itervar_stack_.size()), ann_type,
                             topdown_product_,
                             static_cast<int>(itervar_counter_++))});
  }

  return true;
}

void TouchExtractor::ExitItervar_() {
  if (!skip_stack_size_.empty() &&
      skip_stack_size_.back() == itervar_stack_.size()) {
    skip_stack_size_.pop_back();
    return;
  }
  Var var = itervar_stack_.back();

  // update count and reuse ratio for upper iter vars (includes self)
  for (auto kv : itervar_map[var].touch_feature) {
    if (kv.second.stride != 0) {  // multiply count
      for (auto stack_var : itervar_stack_) {
        auto touch_pattern =
            itervar_map[stack_var].touch_feature.find(kv.first);
        ICHECK(touch_pattern != itervar_map[stack_var].touch_feature.end());
        touch_pattern->second.count *= itervar_map[var].length;
      }
    } else {  // multiply reuse ratio
      for (auto stack_var : itervar_stack_) {
        auto touch_pattern =
            itervar_map[stack_var].touch_feature.find(kv.first);
        ICHECK(touch_pattern != itervar_map[stack_var].touch_feature.end());
        touch_pattern->second.reuse *= itervar_map[var].length;
      }
    }
  }
  itervar_stack_.pop_back();

  int64_t length = itervar_map[var].length;
  if (length != 0) topdown_product_ /= length;
  int64_t bottomup_product = -1;
  for (auto kv : itervar_map[var].touch_feature) {
    bottomup_product =
        std::max(bottomup_product, kv.second.count * kv.second.reuse);
  }

  itervar_map[var].bottomup_product = bottomup_product;

  // push base to upper parallel axis
  int para_level = ParallelLevel(itervar_map[var].ann);
  // if is the separate line of parallel level, push the base to upper parallel
  // level
  if (!itervar_stack_.empty() &&
      ParallelLevel(itervar_map[itervar_stack_.back()].ann) == para_level + 1) {
    for (auto kv : itervar_map[var].touch_feature) {
      for (auto stack_var : itervar_stack_) {
        if (ParallelLevel(itervar_map[stack_var].ann) == para_level + 1) {
          auto touch_pattern =
              itervar_map[stack_var].touch_feature.find(kv.first);
          ICHECK(touch_pattern != itervar_map[stack_var].touch_feature.end());
          // NOTE: use minus as a flag to denote it is a base,
          // indicating it is not the final value
          touch_pattern->second.thread_reuse = -kv.second.reuse;
          touch_pattern->second.thread_count = -kv.second.count;
        }
      }
    }
  }

  for (auto kv : itervar_map[var].touch_feature) {
    if (kv.second.thread_count < 0) {
      itervar_map[var].touch_feature[kv.first].thread_count =
          kv.second.count / (-kv.second.thread_count);
      itervar_map[var].touch_feature[kv.first].thread_reuse =
          kv.second.reuse / (-kv.second.thread_reuse);
    }
  }
}

void TouchExtractor::EnterMem_(Var buffer_var, PrimExpr index) {
  std::string name = buffer_var.get()->name_hint;
  TouchedBuffer buf = name + "_" + std::to_string(buffer_counter_[name]++);

  // extract touch pattern from index
  IndexParser parser;
  parser.Parse(index);

  // push up mem access info
  for (auto var : itervar_stack_) {
    auto x = parser.pattern_map.find(var.get());
    if (x != parser.pattern_map.end()) {
      itervar_map[var].touch_feature[buf] = x->second;
    } else {
      itervar_map[var].touch_feature[buf] = TouchPattern();
    }
  }
}

// serialize a tree
int DFSSerialize(std::shared_ptr<const Tree> root,
                 std::vector<std::vector<int>> *children,
                 std::vector<std::string> *names,
                 std::vector<std::vector<float>> *additionals) {
  std::vector<int> node_children;
  for (auto child : root->children) {
    int child_id = DFSSerialize(child, children, names, additionals);
    node_children.push_back(child_id);
  }

  int idx = static_cast<int>(children->size());
  children->push_back(node_children);
  names->push_back(root->name);
  additionals->push_back(root->additional);

  return idx;
}

void GetLSTMFeature(const Stmt& stmt, int cache_line_size, bool add_stats,
                    std::vector<char> *data) {
  std::shared_ptr<Tree> root = std::make_shared<Tree>("root");
  ASTExtractor extractor;

  if (add_stats) {
    TouchExtractor touch_ext;

    // extract touch feature
    touch_ext.Analyze(stmt);

    // sort loop vars according to order
    std::vector<tir::Var> vars;
    for (auto kv : touch_ext.itervar_map) {
      vars.push_back(kv.first);
    }
    std::sort(vars.begin(), vars.end(),
              [&](const tir::Var &lhs, const tir::Var &rhs)
                  -> bool {
      return
          touch_ext.itervar_map[lhs].order < touch_ext.itervar_map[rhs].order;
    });

    // find maximum depth of loop nests and the innermost buffers
    int max_depth = 0;
    std::set<std::string> added;
    std::set<TouchedBuffer> innermost_buffers;

    for (auto var : vars) {
      ItervarFeature &fea = touch_ext.itervar_map[var];
      max_depth = std::max(max_depth, fea.nest_level);
    }

    // mark inner most buffer
    for (auto iter = vars.rbegin(); iter != vars.rend(); iter++) {
      auto var = *iter;
      ItervarFeature &fea = touch_ext.itervar_map[var];
      if (fea.nest_level == max_depth) {
        for (auto kv : fea.touch_feature) {
          std::string raw_name = kv.first.substr(0, kv.first.size() - 2);
          size_t pos = raw_name.find(".");
          if (pos < kv.first.size())
            raw_name = raw_name.substr(0, pos);

          if (added.find(raw_name) == added.end()) {
            innermost_buffers.insert(kv.first);
            added.insert(raw_name);
          }
        }
      }
    }

    extractor.Extract(stmt, root, &touch_ext.itervar_map, &innermost_buffers);
  } else {
    extractor.Extract(stmt, root, nullptr, nullptr);
  }

  // serialize tree structure for front end
  std::vector<std::vector<int>> children;
  std::vector<std::string> names;
  std::vector<std::vector<float>> additionals;
  DFSSerialize(root, &children, &names, &additionals);

  // calculate size
  int32_t n_tree = static_cast<int>(children.size());
  int32_t offset_child, offset_name, offset_additional;
  int32_t nbytes_child, nbytes_name, nbytes_add;
  int32_t total_size;

  nbytes_child = nbytes_name = nbytes_add = n_tree * sizeof(int32_t);
  for (int i = 0; i < n_tree; i++) {
    nbytes_child += children[i].size() * sizeof(int32_t);
    nbytes_name += names[i].size() * sizeof(char);
    nbytes_add += additionals[i].size() * sizeof(float);
  }

  offset_child = sizeof(int32_t) * 4;
  offset_name = offset_child + nbytes_child;
  offset_additional = offset_name + nbytes_name;
  total_size = offset_additional + nbytes_add;

  // serialize to bytes
  data->resize(static_cast<size_t>(total_size), 0);
  char *pdata = data->data();
  int32_t header[] = {n_tree, offset_child, offset_name, offset_additional};

  memcpy(pdata, header, sizeof(header));
  int32_t ct, num;

  ct = 0;
  for (int i = 0; i < n_tree; i++) {
    num = static_cast<int32_t>(children[i].size());
    memcpy(pdata + offset_child + sizeof(num) * i, &num, sizeof(num));
    memcpy(pdata + offset_child + sizeof(num) * n_tree + ct * sizeof(int32_t),
           children[i].data(), num * sizeof(int32_t));
    ct += num;
  }

  ct = 0;
  for (int i = 0; i < n_tree; i++) {
    num = static_cast<int32_t>(names[i].size());
    memcpy(pdata + offset_name + sizeof(num) * i, &num, sizeof(num));
    memcpy(pdata + offset_name + sizeof(num) * n_tree + ct * sizeof(int8_t),
           names[i].data(), num * sizeof(int8_t));
    ct += num;
  }

  ct = 0;
  for (int i = 0; i < n_tree; i++) {
    num = static_cast<int32_t>(additionals[i].size());
    memcpy(pdata + offset_additional + sizeof(num) * i, &num, sizeof(num));
    memcpy(pdata + offset_additional + sizeof(num) * n_tree +
               ct * sizeof(float),
           additionals[i].data(), num * sizeof(float));
    ct += num;
  }
}

void GetLSTMFeatureFromState(const SearchTask& task, const State& state,
                             std::vector<char>* feature,
                             std::atomic<int>* error_ct) {
  te::Schedule sch;
  Array<te::Tensor> tensors;

  std::tie(sch, tensors) = task->compute_dag.ApplySteps(state->transform_steps);

  // When inlining, replace const matrices with const values.
  // Produces wrong IR, but good enough for feature extraction, and
  // can improve the speed of feature extraction/search.  Must be
  // called before ScheduleToModule to have an effect.
  sch = sch.normalize_for_feature_extraction();

  try {
    const std::string& name = "main";
    auto pass_ctx = tvm::transform::PassContext::Current();

    auto mod =
        ScheduleToModule(sch, Array<ObjectRef>{tensors.begin(), tensors.end()},
                         name, std::unordered_map<te::Tensor, te::Buffer>());

    bool disable_vectorize =
        pass_ctx->GetConfig<Bool>("tir.disable_vectorize", Bool(false)).value();
    bool instrument_bound_checkers =
        pass_ctx->GetConfig<Bool>(
            "tir.instrument_bound_checkers", Bool(false)).value();

    if (IsGPUTask(task)) {
      auto pass_list = Array<tvm::transform::Pass>();
      // Phase 0
      pass_list.push_back(tir::transform::InjectPrefetch());
      pass_list.push_back(
          tir::transform::StorageFlatten(64, instrument_bound_checkers));
      // Phase 1
      pass_list.push_back(tir::transform::NarrowDataType(32));
      pass_list.push_back(tir::transform::Simplify());
      pass_list.push_back(tir::transform::VectorizeLoop(!disable_vectorize));
      pass_list.push_back(tir::transform::InjectVirtualThread());
      pass_list.push_back(tir::transform::StorageRewrite());
      pass_list.push_back(tir::transform::Simplify());
      tvm::Map<String, tvm::PrimExpr> gpu_params{
          {"max_shared_memory_per_block",
            task->hardware_params->max_shared_memory_per_block},
          {"max_local_memory_per_block",
            task->hardware_params->max_local_memory_per_block},
          {"max_threads_per_block",
            task->hardware_params->max_threads_per_block},
          {"max_vector_bytes",
            task->hardware_params->vector_unit_bytes},
          {"max_vthread",
            task->hardware_params->max_vthread_extent},
      };
      pass_list.push_back(tir::transform::VerifyGPUCode(gpu_params));
      const auto& optimize = tir::transform::Sequential(pass_list);
      optimize(mod);
    }
    const auto& optimize =
        tir::transform::Sequential(
            Array<tvm::transform::Pass>{tir::transform::Simplify()});
    mod = optimize(std::move(mod));
    PrimFunc prim_func = Downcast<PrimFunc>(mod->Lookup(name));
    GetLSTMFeature(
        /* stmt = */prim_func->body,
        /* cahce_line_size = */task->hardware_params->cache_line_bytes,
        /* add_stats = */true,
        /* data = */ feature);
  } catch (Error& e) {
    if (error_ct != nullptr) {
      (*error_ct)++;
    }
  }
}

void GetLSTMFeaturesFromStates(
    const SearchTask& task, const Array<State>& states,
    int skip_first_n_feature_extraction,
    std::vector<std::vector<char>>* features) {
  // extract features
  features->assign(states.size(), std::vector<char>());
  std::atomic<int> error_ct(0);

  support::parallel_for(
      skip_first_n_feature_extraction, states.size(),
      [&task, &states, &features, &error_ct](int i) {
        GetLSTMFeatureFromState(task, states[i], &(*features)[i], &error_ct);
      });
}

/*
 * \brief Serialize a two-dimensional variable-size feature vector with
 * normalized throughputs and task ids to a one-dimensional flatten byte array.
 *
 * For faster data copy between c++ and python, the c++ part returns features in
 * a single flatten array using a packed format. The python part then unpacks
 * the flatten array.
 *
 * The packed format for n records is:
 * {
 *   int   n;
 *   int   sizes[n+2];           // The sizes for the following arrays
 *
 *   float features_0[size[0]];  // The features for record 0
 *   float features_1[size[1]];  // The features for record 1
 *   ...
 *   float features_i[size[i]];  // The features for record i
 *   ... // until i == n - 1
 *
 *   float throughputs[sizes[n]];  // The normalized throughputs for n records
 *   int   task_ids[size[n+1]];   // The task ids for n records
 * }
 * To implement this format, we also store int as float, so we can store all
 * numbers into a single float array.
 */
TVMByteArray SerializeFeatures(std::vector<std::vector<char>>&& features,
                               std::vector<float>&& normalized_throughputs,
                               std::vector<int>&& task_ids,
                               std::vector<char>* out_data) {
  size_t total_bytes = 0;
  std::vector<int> size_vector;

  int n = features.size();

  // serialize sizes
  size_t size_vector_size = 1 + n + 2;
  total_bytes += size_vector_size * sizeof(int);

  size_vector.reserve(size_vector_size);
  size_vector.push_back(features.size());
  for (const auto& x : features) {
    size_vector.push_back(static_cast<int>(x.size()));
    total_bytes += x.size();
  }
  size_vector.push_back(static_cast<int>(normalized_throughputs.size()));
  total_bytes += sizeof(float) * normalized_throughputs.size();
  size_vector.push_back(static_cast<int>(task_ids.size()));
  total_bytes += sizeof(int) * task_ids.size();

  ICHECK_EQ(size_vector.size(), size_vector_size);

  // allocate memory
  out_data->reserve(total_bytes);
  char* ptr = out_data->data();

  // serialize size_vector
  memmove(ptr, reinterpret_cast<char*>(size_vector.data()),
          size_vector.size() * sizeof(int));
  ptr += size_vector.size() * sizeof(int);

  // serialize features
  for (auto& x : features) {
    memmove(ptr, x.data(), x.size());
    ptr += x.size();
    x.clear();
  }

  // serialize normalized_throughputs
  memmove(ptr, reinterpret_cast<char*>(normalized_throughputs.data()),
          normalized_throughputs.size() * sizeof(int));
  ptr += normalized_throughputs.size() * sizeof(int);

  // serialize task_ids
  memmove(ptr, reinterpret_cast<char*>(task_ids.data()),
          task_ids.size() * sizeof(int));
  ptr += task_ids.size() * sizeof(int);

  ICHECK_EQ(ptr - out_data->data(), total_bytes);

  return TVMByteArray{out_data->data(), total_bytes};
}


// Return feature buffer given the state and search task.
TVM_REGISTER_GLOBAL("auto_scheduler.GetLSTMFeatureFromState")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      SearchTask task = args[0];
      State state = args[1];
      std::vector<char> features;
      GetLSTMFeatureFromState(task, state, &features, /* error_ct = */nullptr);
      *ret = TVMByteArray{features.data(), features.size()};
    });

// Return feature buffer given array of states and search task.
TVM_REGISTER_GLOBAL("auto_scheduler.GetLSTMFeaturesFromStates")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      SearchTask task = args[0];
      Array<State> states = args[1];

      std::vector<std::vector<char>> features;
      std::vector<float> normalized_throughputs;
      std::vector<int> task_ids;

      GetLSTMFeaturesFromStates(task, states, 0, &features);

      std::vector<char> byte_data;
      *ret = SerializeFeatures(std::move(features),
                               std::move(normalized_throughputs),
                               std::move(task_ids), &byte_data);
    });

}  // namespace auto_scheduler
}  // namespace tvm
