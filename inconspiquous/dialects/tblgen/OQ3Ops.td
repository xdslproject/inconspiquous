//===- OQ3Ops.td - OpenQASM 3 dialect ops ------------------*- tablegen -*-===//
//
// Copyright 2023 IBM
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This is the main operation definition file for OpenQASM 3 operations.
///
//===----------------------------------------------------------------------===//

#ifndef OQ3_OPS
#define OQ3_OPS

// TODO: Temporary, until constraints between `OpenQASM3`, `QUIR`, `Pulse`, and
// `System` dialects are ironed out.
include "Dialect/QUIR/IR/QUIRInterfaces.td"
include "Dialect/QUIR/IR/QUIRTraits.td"
include "Dialect/QUIR/IR/QUIRTypeConstraints.td"

include "Dialect/OQ3/IR/OQ3Base.td"
include "Dialect/OQ3/IR/OQ3ArithmeticOps.td"
include "Dialect/OQ3/IR/OQ3AngleOps.td"
include "Dialect/OQ3/IR/OQ3ArrayOps.td"
include "Dialect/OQ3/IR/OQ3CastOps.td"
include "Dialect/OQ3/IR/OQ3CBitOps.td"
include "Dialect/OQ3/IR/OQ3GateOps.td"
include "Dialect/OQ3/IR/OQ3VariableOps.td"

#endif // OQ3_OPS
