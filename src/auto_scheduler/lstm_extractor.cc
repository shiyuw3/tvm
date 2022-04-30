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

// START Helper Functions.
int ParallelLevel(AnnotationType ann) {
  switch (ann) {
    case kBlockX:
    case kBlockY:
    case kBlockZ:
      return 2;
    case kThreadX:
    case kThreadY:
    case kThreadZ:
    case kParallel:
      return 1;
    default:
      return 0;
  }
}

/*!
* \brief Return whether the string `value` ends with string `ending`
* \param value Base string
* \param ending Ending string
*/
inline bool EndsWith(std::string const & value, std::string const & ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}
// END Helper Functions.


// START FeatureVisitor.
//
// for loop
void FeatureVisitor::VisitStmt_(const ForNode* op) {
  const auto* extent = op->extent.as<IntImmNode>();
  int64_t loop_extent = -1;
  if (extent != nullptr) {
    loop_extent = extent->value;
  }
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

void FeatureVisitor::VisitExpr_(const BufferLoadNode* op) {
  for (auto index : op->indices) {
    EnterMem_(op->buffer.get()->data, index);
  }
  StmtExprVisitor::VisitExpr_(op);
  ExitMem_();
}

void FeatureVisitor::VisitExpr_(const ProducerLoadNode* op) {
  LOG(FATAL) << "FeatureVisitor for ProducerLoadNode not yet supported";
}

void FeatureVisitor::VisitStmt_(const StoreNode* op) {
  EnterMem_(op->buffer_var, op->index);
  StmtExprVisitor::VisitStmt_(op);
  ExitMem_();
}

void FeatureVisitor::VisitStmt_(const BufferStoreNode* op) {
  for (auto index : op->indices) {
    EnterMem_(op->buffer.get()->data, index);
  }
  StmtExprVisitor::VisitStmt_(op);
  ExitMem_();
}

void FeatureVisitor::VisitStmt_(const ProducerStoreNode* op) {
  LOG(FATAL) << "FeatureVisitor for ProducerStoreNode not yet supported";
}
// END FeatureVisitor.


// START TouchExtractor.
//
// extract iter vars and their touch pattern from ir
bool TouchExtractor::EnterItervar_(
    Var var, int64_t length, AnnotationType ann_type) {
#if 0
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
#endif

  // Use static index to ensure that there are no duplicate var names.
  std::string var_name = var.get()->name_hint;
  std::string new_name =
      var_name + "_" + std::to_string(var_counter_[var_name]++);
  itervar_stack_.push_back(Var(new_name));
  topdown_product_ *= length;
  itervar_map.insert(
      {new_name,
       ItervarFeature(length, static_cast<int>(itervar_stack_.size()), ann_type,
                      topdown_product_,
                      static_cast<int>(itervar_counter_++))});

  return true;
}

void TouchExtractor::ExitItervar_() {
  if (!skip_stack_size_.empty() &&
      skip_stack_size_.back() == itervar_stack_.size()) {
    skip_stack_size_.pop_back();
    return;
  }
  Var var = itervar_stack_.back();
  std::string var_name = var.get()->name_hint;

  // update count and reuse ratio for upper iter vars (includes self)
  for (auto kv : itervar_map[var_name].touch_feature) {
    if (kv.second.stride != 0) {  // multiply count
      for (auto stack_var : itervar_stack_) {
        std::string stack_var_name = stack_var.get()->name_hint;
        auto touch_pattern =
            itervar_map[stack_var_name].touch_feature.find(kv.first);
        ICHECK(touch_pattern !=
                   itervar_map[stack_var_name].touch_feature.end());
        touch_pattern->second.count *= itervar_map[var_name].length;
      }
    } else {  // multiply reuse ratio
      for (auto stack_var : itervar_stack_) {
        std::string stack_var_name = stack_var.get()->name_hint;
        auto touch_pattern =
            itervar_map[stack_var_name].touch_feature.find(kv.first);
        ICHECK(touch_pattern != itervar_map[stack_var_name].touch_feature.end());
        touch_pattern->second.reuse *= itervar_map[var_name].length;
      }
    }
  }
  itervar_stack_.pop_back();

  int64_t length = itervar_map[var_name].length;
  if (length != 0) topdown_product_ /= length;
  int64_t bottomup_product = -1;
  for (auto kv : itervar_map[var_name].touch_feature) {
    bottomup_product =
        std::max(bottomup_product, kv.second.count * kv.second.reuse);
  }

  itervar_map[var_name].bottomup_product = bottomup_product;

  // push base to upper parallel axis
  int para_level = ParallelLevel(itervar_map[var_name].ann);
  // if is the separate line of parallel level, push the base to upper parallel
  // level
  if (!itervar_stack_.empty() &&
      ParallelLevel(itervar_map[itervar_stack_.back().get()->name_hint].ann) ==
          para_level + 1) {
    for (auto kv : itervar_map[var_name].touch_feature) {
      for (auto stack_var : itervar_stack_) {
        std::string stack_var_name = stack_var.get()->name_hint;
        if (ParallelLevel(itervar_map[stack_var_name].ann) == para_level + 1) {
          auto touch_pattern =
              itervar_map[stack_var_name].touch_feature.find(kv.first);
          ICHECK(touch_pattern !=
                     itervar_map[stack_var_name].touch_feature.end());
          // NOTE: use minus as a flag to denote it is a base,
          // indicating it is not the final value
          touch_pattern->second.thread_reuse = -kv.second.reuse;
          touch_pattern->second.thread_count = -kv.second.count;
        }
      }
    }
  }

  for (auto kv : itervar_map[var_name].touch_feature) {
    if (kv.second.thread_count < 0) {
      itervar_map[var_name].touch_feature[kv.first].thread_count =
          kv.second.count / (-kv.second.thread_count);
      itervar_map[var_name].touch_feature[kv.first].thread_reuse =
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
    std::string var_name = var.get()->name_hint;
    auto x = parser.pattern_map.find(var.get());
    if (x != parser.pattern_map.end()) {
      itervar_map[var_name].touch_feature[buf] = x->second;
    } else {
      itervar_map[var_name].touch_feature[buf] = TouchPattern();
    }
  }
}
// END TouchExtractor.


// START ASTExtractor.
//
// Extract function.
void ASTExtractor::Extract(
    Stmt stmt, std::shared_ptr<Tree> root,
    const std::unordered_map<std::string, ItervarFeature> *itervar_map,
    const std::set<TouchedBuffer> *innermost_buffers) {
  root_stack_.push_back(root);
  itervar_map_ = itervar_map;
  innermost_buffers_ = innermost_buffers;
  CHECK_EQ(itervar_map == nullptr, innermost_buffers == nullptr);
  this->VisitStmt(stmt);
}

bool ASTExtractor::EnterItervar_(Var var, int64_t length,
                                 AnnotationType ann_type) {
// FIXME(wsy): Seems happen if we count BufferLoad and BufferStore.
#if 0
  if (EndsWith(var.get()->name_hint, ".init")) {
    LOG(FATAL) << "Should never happen!!";
    return false;
  }
#endif
  std::shared_ptr<Tree> node = std::make_shared<Tree>("for");

  // do not attach statistic feature on tree node
  if (itervar_map_ == nullptr) {
    // length
    node->additional.push_back(static_cast<float>(length));
    // one hot annotation
    for (int i = 0; i < kNum; i++) {
      node->additional.push_back(static_cast<float>(i == ann_type));
    }
  } else {
    const ItervarFeature *touch_fea =
        &itervar_map_->find(var.get()->name_hint)->second;

    // check if it is in the longest chain of the tree
    bool found = false;
    for (auto x : touch_fea->touch_feature) {
      if (innermost_buffers_->find(x.first) != innermost_buffers_->end()) {
        found = true;
        break;
      }
    }
    // if it is not in the longest chain of the tree, skip this subtree
    if (!found) return false;

    // length
    node->additional.push_back(static_cast<float>(length));
    // one hot annotation
    for (int i = 0; i < kNum; i++) {
      node->additional.push_back(static_cast<float>(i == ann_type));
    }
    // buffer access patten
    node->additional.push_back(
        static_cast<float>(touch_fea->topdown_product));
    for (auto x : touch_fea->touch_feature) {
      if (innermost_buffers_->find(x.first) == innermost_buffers_->end())
        continue;
      node->additional.push_back(static_cast<float>(x.second.count));
      node->additional.push_back(static_cast<float>(x.second.reuse));
    }
  }
  // add itervar as child
  node->children.push_back(std::make_shared<Tree>(var));

  root_stack_.back()->children.push_back(node);
  root_stack_.push_back(node);
  return true;
}

void ASTExtractor::EnterMem_(Var buffer_var, PrimExpr index) {
#if 0
  if (itervar_map_ != nullptr)
    return;
#endif

  std::shared_ptr<Tree> node = std::make_shared<Tree>(buffer_var);
  IndexvarCollector collector;
  collector.Collect(index);

  for (const VarNode *op : collector.vars)
    node->children.push_back(std::make_shared<Tree>(op->name_hint));

  for (auto iter = root_stack_.rbegin(); iter != root_stack_.rend(); iter++) {
    // attach to nearest loop father node
    if (iter->get()->name == "for") {
      iter->get()->children.push_back(node);
      break;
    }
  }

  root_stack_.push_back(node);
}
// END ASTExtractor.


// START ComputeTensorExtractor.
void ComputeTensorExtractor::Extract(
    Stmt stmt, std::shared_ptr<Tree> root,
    const std::map<std::string, std::string> *buf2name,
    const std::unordered_map<std::string, ItervarFeature> *itervar_map) {
  root_stack_.push_back(root);
  buf2name_ = buf2name;
  itervar_map_ = itervar_map;
  CHECK_EQ(itervar_map == nullptr, buf2name == nullptr);
  this->VisitStmt(stmt);
}

bool ComputeTensorExtractor::EnterItervar_(Var var, int64_t length,
                                           AnnotationType ann_type) {
  if (itervar_map_ == nullptr) {
    LOG(FATAL) << "itervar_map_ of LoopTensorExtractor should not be nullptr";
    return false;
  }

  std::string var_name = var.get()->name_hint;
  std::string new_name =
      var_name + "_" + std::to_string(var_counter_[var_name]++);

  auto touch_fea_iter = itervar_map_->find(new_name);
  if (touch_fea_iter == itervar_map_->end()) {
    LOG(FATAL) << "Var not found in itervar_map_!";
    return false;
  }

  std::shared_ptr<Tree> node = std::make_shared<Tree>(new_name);
  const ItervarFeature *touch_fea = &touch_fea_iter->second;

  // length
  node->additional.push_back(static_cast<float>(touch_fea->length));
  // nest level
  node->additional.push_back(static_cast<float>(touch_fea->nest_level));
  // topdown product.
  node->additional.push_back(static_cast<float>(touch_fea->topdown_product));
  // bottomup product.
  node->additional.push_back(static_cast<float>(touch_fea->bottomup_product));
  // one hot annotation
  for (int i = 0; i < kNum; i++) {
    node->additional.push_back(static_cast<float>(i == touch_fea->ann));
  }

  // add itervar as child
  node->children.push_back(std::make_shared<Tree>(new_name));

  root_stack_.back()->children.push_back(node);
  root_stack_.push_back(node);

  return true;
}

void ComputeTensorExtractor::ExitItervar_() {
  root_stack_.pop_back();
}

void ComputeTensorExtractor::EnterMem_(Var buffer_var, PrimExpr index) {
  std::string name = buffer_var.get()->name_hint;
  TouchedBuffer buf = name + "_" + std::to_string(buffer_counter_[name]++);

  std::shared_ptr<Tree> node = std::make_shared<Tree>(buf);
  IndexvarCollector collector;
  collector.Collect(index);

  for (const VarNode *op : collector.vars)
    node->children.push_back(std::make_shared<Tree>(op->name_hint));

  for (auto iter = root_stack_.rbegin(); iter != root_stack_.rend(); iter++) {
    // Attach to nearest loop father node.
    auto name_iter = buf2name_->find(buf);
    ICHECK(name_iter != buf2name_->end());
    if (iter->get()->name == name_iter->second) {
      // Check whether this buffer has been pushed already.
      bool pushed = false;
      for (auto child : iter->get()->children) {
        std::string buf_name = child.get()->name;
        std::string raw_name = buf_name.substr(0, buf_name.find("_"));
        // Compare raw buffer name.
        if (raw_name == buf.substr(0, buf.find("_"))) {
          pushed = true;
          break;
        }
      }
      // Push back the node if it has not been pushed.
      if (!pushed) {
        iter->get()->children.push_back(node);
      }
      break;
    }
  }

  root_stack_.push_back(node);
}

void ComputeTensorExtractor::ExitMem_() {
  root_stack_.pop_back();
}
// END ComputeTensorExtractor.


// START LoopTensorExtractor.
void LoopTensorExtractor::Extract(
    Stmt stmt, std::shared_ptr<Tree> root,
    const std::unordered_map<std::string, ItervarFeature> *itervar_map) {
  root_stack_.push_back(root);
  itervar_map_ = itervar_map;
  this->VisitStmt(stmt);
}

bool LoopTensorExtractor::EnterItervar_(Var var, int64_t length,
                                        AnnotationType ann_type) {
  if (itervar_map_ == nullptr) {
    LOG(FATAL) << "itervar_map_ of LoopTensorExtractor should not be nullptr";
    return false;
  }

  std::string var_name = var.get()->name_hint;
  std::string new_name =
      var_name + "_" + std::to_string(var_counter_[var_name]++);

  auto touch_fea_iter = itervar_map_->find(new_name);
  if (touch_fea_iter == itervar_map_->end()) {
    LOG(FATAL) << "Var not found in itervar_map_!";
    return false;
  }

  std::shared_ptr<Tree> node = std::make_shared<Tree>(new_name);
  const ItervarFeature *touch_fea = &touch_fea_iter->second;

  // length
  node->additional.push_back(static_cast<float>(touch_fea->length));
  // nest level
  node->additional.push_back(static_cast<float>(touch_fea->nest_level));
  // topdown product.
  node->additional.push_back(static_cast<float>(touch_fea->topdown_product));
  // bottomup product.
  node->additional.push_back(static_cast<float>(touch_fea->bottomup_product));
  // one hot annotation
  for (int i = 0; i < kNum; i++) {
    node->additional.push_back(static_cast<float>(i == touch_fea->ann));
  }

  // add itervar as child
  node->children.push_back(std::make_shared<Tree>(new_name));

  root_stack_.back()->children.push_back(node);
  root_stack_.push_back(node);

  return true;
}

void LoopTensorExtractor::ExitItervar_() {
  root_stack_.pop_back();
}
// END LoopTensorExtractor.


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
  std::shared_ptr<Tree> cte_root = std::make_shared<Tree>("cte_root");
  std::shared_ptr<Tree> lte_root = std::make_shared<Tree>("lte_root");
  // ASTExtractor extractor;
  ComputeTensorExtractor cte;
  LoopTensorExtractor lte;

  if (add_stats) {
    TouchExtractor touch_ext;

    // extract touch feature
    touch_ext.Analyze(stmt);

    // sort loop vars according to order
    std::vector<std::string> var_names;
    for (auto kv : touch_ext.itervar_map) {
      var_names.push_back(kv.first);
    }
    std::sort(var_names.begin(), var_names.end(),
              [&](const std::string &lhs, const std::string &rhs) -> bool {
      return
          touch_ext.itervar_map[lhs].order < touch_ext.itervar_map[rhs].order;
    });

    // Find the touched buffers and corresponding innermost levels.
    std::map<TouchedBuffer, int> buf2level;
    std::map<TouchedBuffer, std::string> buf2name;
    for (auto kv : touch_ext.itervar_map) {
      ItervarFeature fea = kv.second;
      for (auto touch_fea : fea.touch_feature) {
        if (buf2level[touch_fea.first] < fea.nest_level) {
          buf2level[touch_fea.first] = fea.nest_level;
          buf2name[touch_fea.first] = kv.first;
        }
      }
    }

    // Extract compute tensor.
    cte.Extract(stmt, cte_root, &buf2name, &touch_ext.itervar_map);
    // Extract loop tensor.
    lte.Extract(stmt, lte_root, &touch_ext.itervar_map);
  } else {
    // Extract compute tensor.
    cte.Extract(stmt, cte_root, nullptr, nullptr);
    // Extract loop tensor.
    lte.Extract(stmt, lte_root, nullptr);
  }

  // serialize tree structure for front end
  std::vector<std::vector<int>> children;
  std::vector<std::string> names;
  std::vector<std::vector<float>> additionals;
  DFSSerialize(cte_root, &children, &names, &additionals);

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
 *        normalized throughputs and task ids to a one-dimensional flatten byte
 *        array.
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
