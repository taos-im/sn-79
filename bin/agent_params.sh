#!/usr/bin/env bash
# Per-agent launch parameters, in ONE place.
#
# Agents read required settings off self.config, sourced from `--agent.params k=v ...` at launch. A
# missing parameter is not a soft failure: initialize() raises AttributeError and the miner process
# exits, so every launcher needs the same map and it is shared rather than copied.
#
#   source bin/agent_params.sh
#   params=$(agent_params FuturesAgent)
#   ... --agent.params $params        # deliberately unquoted: each k=v is its own argv item

agent_params() {
  case "$1" in
    RandomTakerAgent)   echo "min_quantity=0.1 max_quantity=1.0 expiry_period=200 max_fee_rate=0.01 min_leverage=0.0 max_leverage=0.0 gtx_training_enabled=true gtx_collect_data=true" ;;
    RandomMakerAgent)   echo "min_quantity=0.1 max_quantity=1.0 expiry_period=200 max_fee_rate=0.01 min_leverage=0.0 max_leverage=0.0" ;;
    ImbalanceAgent)     echo "imbalance_depth=5 history_retention_mins=1 expiry_period=200 parallel_history_workers=1 gtx_training_enabled=true gtx_collect_data=true" ;;
    OrderOptionAgent)   echo "min_quantity=0.1 max_quantity=1.0 gtx_training_enabled=true gtx_collect_data=true" ;;
    SelfTradingAgent)   echo "min_quantity=0.1 max_quantity=1.0" ;;
    FuturesAgent)       echo "quantity=0.5 expiry_period=200 sampling_period=1000000000" ;;
    SLTPAgent)          echo "quantity=0.5" ;;
    SimpleRegressorAgent) echo "model=PassiveAggressiveRegressor quantity=0.5 signal_threshold=0.0025 model_threshold=0.5 expiry_period=200" ;;
    MovingHurstAgent)   echo "gtx_training_enabled=true gtx_collect_data=true" ;;
    DevAgent)           echo "gtx_training_enabled=true gtx_collect_data=true" ;;
    RevengAgent)        echo "gtx_training_enabled=true gtx_collect_data=true" ;;
    CustomTrainingAgent) echo "gtx_training_enabled=true gtx_collect_data=true" ;;
    HybridTrainingAgent) echo "imbalance_depth=5 history_retention_mins=1 entry_threshold=0.35 cancel_threshold=0.20 stop_loss_bps=40 base_quote_size=0.3 enter_size_mult=3.0 max_flat_inventory=2.0 expiry_period=500000000 max_fee_rate=0.005" ;;
    *) echo "" ;;
  esac
}
