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

#ifndef TVM_AUTO_SCHEDULER_LSTM_EXTRACTOR_H_
#define TVM_AUTO_SCHEDULER_LSTM_EXTRACTOR_H_

#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace auto_scheduler {

using namespace tvm::tir;

/*!
 * \brief Type of for loop, used as one-hot encoding in features
 */
enum AnnotationType {
  kBlockX,
  kBlockY,
  kBlockZ,
  kThreadX,
  kThreadY,
  kThreadZ,
  kUnrolled,
  kVectorized,
  kParallel,
  kSerial,
  kVirtualThread,
  kNum,
};

/*!
 * \brief A base class for feature extractor, used for processing
 * for loop and memory access in the IR
 */
class FeatureVisitor : public StmtExprVisitor {
 public:
  // for loop
  void VisitStmt_(const ForNode* op) final;
  void VisitStmt_(const AttrStmtNode* op) final;

  // memory access
  void VisitExpr_(const LoadNode* op) final;
  void VisitExpr_(const BufferLoadNode* op) final;
  void VisitExpr_(const ProducerLoadNode* op) final;
  void VisitStmt_(const StoreNode* op) final;
  void VisitStmt_(const BufferStoreNode* op) final;
  void VisitStmt_(const ProducerStoreNode* op) final;

  using StmtExprVisitor::VisitExpr_;
  using StmtExprVisitor::VisitStmt_;

 protected:
  /*!
   * \brief Enter a for loop node
   * \param var The expression to be printed.
   * \param length The output stream
   * \param ann_type The type for the for loop
   * \return skip Whether skip this node
   */
  virtual bool EnterItervar_(tir::Var var, int64_t length,
                             AnnotationType ann_type) = 0;
  /*! \brief Exit a for loop subtree */
  virtual void ExitItervar_() = 0;
  /*!
   * \brief Enter a memory access node
   * \param buffer_var The buffer to access.
   * \param index Index expression
   */
  virtual void EnterMem_(tir::Var buffer_var, tvm::PrimExpr index) = 0;
  /*! \brief Exit a memory access node */
  virtual void ExitMem_() = 0;
};

using TouchedBuffer = std::string;
// touch pattern buf[(stride * var) % mod) + other]
struct TouchPattern {
  int64_t stride{0};
  int64_t mod{-1};  // -1 for +inf

  int64_t count{1};
  int64_t reuse{1};
  int64_t thread_count{0};  // count when move thread axis into innermost
  int64_t thread_reuse{0};  // reuse ratio move thread axis into innermost
};

// all the feature of an iter var
struct ItervarFeature {
  ItervarFeature(Var var, int64_t extent, int nest, AnnotationType ann_type,
                 int64_t topdown, int counter)
      : length(extent), nest_level(nest), ann(ann_type),
        topdown_product(topdown), order(counter) {}
  ItervarFeature() {}

  // Axis Attributes
  int64_t length{0};
  int nest_level{0};
  // one-hot axis type
  AnnotationType ann;
  // accumulative product of axis length, in top-down order
  int64_t topdown_product{0};
  // accumulative product of axis length, in bottom-up order
  // bottomup_product = reuse * count for any touched buffer
  int64_t bottomup_product{0};
  // used for soring axis
  int order{0};

  // Arithmetic feature
  int add_ct{0};
  int mul_ct{0};
  int div_ct{0};

  // Memory Touch Feature
  std::unordered_map<TouchedBuffer, TouchPattern> touch_feature;
};

// get touch pattern from index expression
class IndexParser : public ExprVisitor {
 public:
  void Parse(PrimExpr expr) {
    pattern_map.clear();
    this->VisitExpr(expr);
  }

  void VisitExpr_(const VarNode* op) final {
    // TODO(lmzheng): handle more index types (multiple occurrence)
    if (pattern_map.count(op) == 0) {
      pattern_map[op] = TouchPattern();
      pattern_map[op].stride = next_stride_;
      next_stride_ = 1;
    }
  }

  void VisitExpr_(const MulNode* op) final {
    if (op->a.as<VarNode>()) {
      if (const auto stride = op->b.as<IntImmNode>()) {
        next_stride_ = stride->value;
      }
    }
    ExprVisitor::VisitExpr_(op);
  }

  std::unordered_map<const VarNode*, TouchPattern> pattern_map;

 private:
  int64_t next_stride_ = 1;
};

// extract iter vars and their touch pattern from ir
class TouchExtractor : public FeatureVisitor {
 public:
  void Analyze(const Stmt& stmt) { operator()(stmt); }

  // arithmetic stats
  void VisitExpr_(const AddNode* op) final {
    if (op->dtype.is_float()) itervar_map[itervar_stack_.back()].add_ct++;
    FeatureVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubNode* op) final {
    if (op->dtype.is_float()) itervar_map[itervar_stack_.back()].add_ct++;
    FeatureVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const MulNode* op) final {
    if (op->dtype.is_float()) itervar_map[itervar_stack_.back()].mul_ct++;
    FeatureVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const DivNode* op) final {
    if (op->dtype.is_float()) itervar_map[itervar_stack_.back()].div_ct++;
    FeatureVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const ModNode* op) final {
    if (op->dtype.is_float()) itervar_map[itervar_stack_.back()].div_ct++;
    FeatureVisitor::VisitExpr_(op);
  }

  std::unordered_map<Var, ItervarFeature,
                     tvm::ObjectPtrHash, tvm::ObjectPtrEqual> itervar_map;

 private:
  bool EnterItervar_(Var var, int64_t length, AnnotationType ann_type);
  void ExitItervar_();
  void EnterMem_(Var buffer_var, PrimExpr index);
  void ExitMem_() {}

  int64_t topdown_product_{1};
  std::map<std::string, size_t> buffer_counter_;
  size_t itervar_counter_{0};
  // use deque instead of stack for indexing
  std::deque<Var> itervar_stack_;
  std::deque<size_t> skip_stack_size_;

  using FeatureVisitor::VisitExpr_;
};

// a node in AST
class Tree {
 public:
  explicit Tree(Var var) {
    name = var.get()->name_hint;
  }

  explicit Tree(std::string node_name) : name(node_name) {}

  std::string name;
  std::vector<std::shared_ptr<Tree>> children;
  std::vector<float> additional;
};

// collect all index vars from a buffer index
// Note: Since the IndexvarCollector inherited from IRVisitor in original
// implementation, while IRVisitor is removed in current TVM version. We
// chose to follow the behavior of another class IndexParser with the same
// inheritance.
class IndexvarCollector: public ExprVisitor {
 public:
  void Collect(PrimExpr expr) {
    this->VisitExpr(expr);
  }

  void VisitExpr_(const VarNode *op) final {
    vars.insert(op);
  }

  std::set<const VarNode*> vars;
};

// extract simplified ast
class ASTExtractor : public FeatureVisitor {
 public:
  void Extract(Stmt stmt, std::shared_ptr<Tree> root,
               const std::unordered_map<Var, ItervarFeature, tvm::ObjectPtrHash,
                                        tvm::ObjectPtrEqual> *itervar_map,
               const std::set<TouchedBuffer> *innermost_buffers);

 private:
  bool EnterItervar_(Var var, int64_t length, AnnotationType ann_type);
  void ExitItervar_() {
    root_stack_.pop_back();
  }
  void EnterMem_(Var buffer_var, PrimExpr index);
  void ExitMem_() {
#if 0
    if (itervar_map_ != nullptr)
      return;
#endif
    root_stack_.pop_back();
  }

 private:
  std::deque<std::shared_ptr<Tree>> root_stack_;
  const std::unordered_map<Var, ItervarFeature, tvm::ObjectPtrHash,
                           tvm::ObjectPtrEqual> *itervar_map_;
  const std::set<TouchedBuffer> *innermost_buffers_;
};

// Loop tensor extractor.
class LoopTensorExtractor : public FeatureVisitor {
 public:
  void Extract(Stmt stmt, std::shared_ptr<Tree> root,
               const std::unordered_map<Var, ItervarFeature, tvm::ObjectPtrHash,
                                        tvm::ObjectPtrEqual> *itervar_map);

 private:
  // Instantiation of pure functions.
  bool EnterItervar_(Var var, int64_t length, AnnotationType ann_type);
  void ExitItervar_();
  void EnterMem_(Var buffer_var, PrimExpr index) {}
  void ExitMem_() {}

 private:
  std::deque<std::shared_ptr<Tree>> root_stack_;
  const std::unordered_map<Var, ItervarFeature, tvm::ObjectPtrHash,
                           tvm::ObjectPtrEqual> *itervar_map_;
};

void GetLSTMFeature(const Stmt& stmt, bool add_stats, std::vector<char> *data);

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_LSTM_EXTRACTOR_H_
