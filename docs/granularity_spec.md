# Granularity Ablation Specification

## Canonical Nodes

- near_key
- has_key
- near_target
- has_target
- opened_box
- opened_door
- has_box
- at_goal
- task_success

`task_success` is only used as a terminal variable, not as a reusable module node.

## G0: Coarse Predicate Modules

- has_key
- opened_door
- has_target
- opened_box
- has_box
- at_goal

## G1: Predicate-Causal Edge Modules

Atomic Policies:

- opened_box
- near_key
- near_target
- at_goal

Edge Modules:

- near_key -> has_key
- has_key -> opened_door
- near_target -> has_target
- opened_box -> has_key
- opened_door -> has_box
- opened_door -> at_goal

## G2: Action Template Modules

- B1: approach_key
- B2: pickup_key
- B3: approach_door
- B4: toggle_door
- B5: approach_target
- B6: pickup_target
- B7: approach_goal
- B8: toggle_box

## Granularity Selection Tasks

Door chain:

- MiniGrid-DoorKey-6x6-v0
- BabyAI-UnlockLocal-v0
- MiniGrid-UnlockPickup-v0

Pickup chain:

- MiniGrid-Fetch-8x8-N3-v0
- BabyAI-Pickup-v0

Stage1 smoke test passed on five representative tasks:
DoorKey, UnlockLocal, UnlockPickup, Fetch, BabyAI-Pickup.