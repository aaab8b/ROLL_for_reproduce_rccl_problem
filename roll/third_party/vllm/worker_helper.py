import gc
from typing import Tuple, Iterable

import torch
import torch.distributed as dist
import vllm

from roll.utils.collective import collective
from roll.utils.functionals import get_dist_info_from_comm_plan
from roll.utils.logging import get_logger
from roll.utils.send_recv_utils import RecvBucketManager
from roll.third_party.vllm.vllm_utils import patch_vllm_moe_model_weight_loader
from roll.platforms import current_platform
from vllm.model_executor.model_loader.utils import process_weights_after_loading, set_default_torch_dtype
import os
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.model_loader.utils import device_loading_context
from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock, Qwen3MoeDecoderLayer
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.models.utils import sequence_parallel_chunk
from typing import Any, Optional, Union
logger = get_logger()
# import torch.nn.functional as F


# def _maybe_pad_weight(weight: torch.Tensor) -> torch.Tensor:
#     # Pad the weight tensor. This is an optimization on ROCm platform, which
#     # can benefit from tensors located far enough from one another in memory
#     if (
#         weight.stride(-1) == 1
#         and (weight.stride(-2) * weight.element_size()) % 512 == 0
#     ):
#         num_pad = 256 // weight.element_size()
#         weight = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
#         torch.cuda.empty_cache()

#     return weight

class WorkerHelper:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_loaded : bool = True
        self.kv_cache_loaded : bool = True
        self.have_shuffled_weights :bool = True,
        self.buffers = None
        # self.moe_hook_data = {"inputs": [], "outputs": [], "inter_states": []}
    
    # def _create_forward_impl_wrapper(self, module, module_name):
        # """Create a wrapper for forward_impl to capture inputs and outputs.
        
        # This approach directly wraps the forward_impl method which is the actual
        # implementation that processes hidden_states and router_logits.
        # """
        # worker_helper = self  # Capture self reference for the closure
        # def forward_wrapper(
        #     positions: torch.Tensor,
        #     hidden_states: torch.Tensor,
        #     residual: Optional[torch.Tensor],
        # ) -> tuple[torch.Tensor, torch.Tensor]:
        #      #logging
        #     input_data = {
        #         "hidden_states": hidden_states.detach().cpu().clone(),
        #         "positions": positions.detach().cpu().clone(),
        #     }
        #     worker_helper.moe_hook_data["inputs"].append(input_data)
        #     # Self Attention
        #     if residual is None:
        #         residual = hidden_states
        #         hidden_states = module.input_layernorm(hidden_states)
        #     else:
        #         hidden_states, residual = module.input_layernorm(hidden_states, residual)
        #     inter_data = {
        #         "after_ln_hidden_states": hidden_states.detach().cpu().clone(),
        #         "after_ln_residual": residual.detach().cpu().clone() if residual is not None else None,
        #     }
        #     hidden_states = module.self_attn(
        #         positions=positions,
        #         hidden_states=hidden_states,
        #     )
        #     inter_data["after_self_attn_hidden_states"] = hidden_states.detach().cpu().clone()
        #     # Fully Connected
        #     hidden_states, residual = module.post_attention_layernorm(hidden_states, residual)
        #     inter_data["after_post_attention_layernorm_hidden_states"] = hidden_states.detach().cpu().clone()
        #     inter_data["after_post_attention_layernorm_residual"] = residual.detach().cpu().clone() if residual is not None else None,
        #     worker_helper.moe_hook_data["inter_states"].append(inter_data)
        #     hidden_states = module.mlp(hidden_states)
        #     output_data = {
        #         "hidden_states": hidden_states.detach().cpu().clone(),
        #         "residual": residual.detach().cpu().clone() if residual is not None else None,
        #     }
        #     worker_helper.moe_hook_data["outputs"].append(output_data)
        #     worker_helper.save_moe_hook_data()
        #     return hidden_states, residual
        # return forward_wrapper
        # def forward_wrapper(hidden_states: torch.Tensor) -> torch.Tensor:
        #     assert hidden_states.dim() <= 2, (
        #         "Qwen3MoeSparseMoeBlock only supports 1D or 2D inputs"
        #     )
        #     #logging
        #     input_data = {
        #         "hidden_states": hidden_states.detach().cpu().clone(),
        #     }
        #     worker_helper.moe_hook_data["inputs"].append(input_data)
            
        #     is_input_1d = hidden_states.dim() == 1
        #     num_tokens, hidden_dim = hidden_states.shape
        #     hidden_states = hidden_states.view(-1, hidden_dim)

        #     if module.is_sequence_parallel:
        #         hidden_states = sequence_parallel_chunk(hidden_states)
        #     # router_logits: (num_tokens, n_experts)
        #     router_logits, _ = module.gate(hidden_states)
            
        #     final_hidden_states = module.experts(
        #         hidden_states=hidden_states, router_logits=router_logits
        #     )
        #     #logging
        #     inter_data={
        #         "reshaped_input_hidden_states": hidden_states.detach().cpu().clone(),
        #         "router_logits": router_logits.detach().cpu().clone(),
        #         "final_hidden_states_after": final_hidden_states.detach().cpu().clone(),
        #     }
        #     worker_helper.moe_hook_data["inter_states"].append(inter_data)

        #     if module.is_sequence_parallel:
        #         final_hidden_states = tensor_model_parallel_all_gather(
        #             final_hidden_states, 0
        #         )
        #         final_hidden_states = final_hidden_states[:num_tokens]
            
        #     # return to 1d if input is 1d
        #     worker_helper.moe_hook_data["outputs"].append(final_hidden_states.squeeze(0).detach().cpu().clone() if is_input_1d else final_hidden_states.detach().cpu().clone())
        #     worker_helper.save_moe_hook_data()
        #     return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states
        # return forward_wrapper
        # original_forward_impl = module.forward_impl
        # worker_helper = self  # Capture self reference for the closure
        
        # def wrapped_forward_impl(hidden_states, router_logits):
        #     # Capture inputs
        #     input_data = {
        #         "hidden_states": hidden_states.detach().cpu().clone(),
        #         "router_logits": router_logits.detach().cpu().clone(),
        #     }
        #     worker_helper.moe_hook_data["inputs"].append(input_data)
            
        #     logger.info(f"MoE capturing inputs - Module: {module_name}, "
        #                f"Hidden states shape: {hidden_states.shape}, "
        #                f"Router logits shape: {router_logits.shape}")
            
        #     # ========== Start of forward_impl logic from layer.py ==========
        #     # Now you can dump intermediate variables here
        #     assert module.quant_method is not None
            
        #     module.ensure_moe_quant_config()
            
        #     # Route to the chunked forward path using the FlashInfer Cutlass kernel
        #     # only when data parallelism (DP) is enabled.
        #     _use_flashinfer_cutlass_kernels = (
        #         module.dp_size > 1 and module.use_flashinfer_cutlass_kernels
        #     )
            
        #     if (
        #         module.moe_parallel_config.use_pplx_kernels
        #         or module.moe_parallel_config.use_deepep_ll_kernels
        #         or _use_flashinfer_cutlass_kernels
        #     ):
        #         output = module.forward_impl_chunked(hidden_states, router_logits)
        #     else:
        #         from vllm.distributed import get_ep_group
        #         from vllm.forward_context import  get_forward_context
        #         from contextlib import nullcontext
                
        #         do_naive_dispatch_combine: bool = (
        #             module.dp_size > 1
        #             and not module.moe_parallel_config.use_deepep_ht_kernels
        #             and not module.moe_config.use_flashinfer_cutlass_kernels
        #         )
                
        #         # If there are shared experts but we are not using a modular kernel, the
        #         # shared experts must be called here
        #         from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEModularKernel
        #         if (
        #             not isinstance(module.quant_method.fused_experts, FusedMoEModularKernel)
        #             and module.shared_experts is not None
        #         ):
        #             shared_output = module.shared_experts(hidden_states)
        #         else:
        #             shared_output = None
                
        #         ctx = get_forward_context()
        #         sp_ctx = (
        #             ctx.dp_metadata.sp_local_sizes(module.sp_size)
        #             if ctx.dp_metadata
        #             else nullcontext()
        #         )
                
        #         with sp_ctx:
        #             if do_naive_dispatch_combine:
        #                 hidden_states, router_logits = get_ep_group().dispatch(
        #                     hidden_states, router_logits, module.is_sequence_parallel
        #                 )
        #             inter_data={"hidden_states": hidden_states.detach().cpu().clone(), "router_logits": router_logits.detach().cpu().clone()}
        #             inter_data["hidden_states_contiguous"]=hidden_states.is_contiguous()
        #             inter_data["hidden_states_stride"]=hidden_states.stride()
        #             inter_data["router_logits_contiguous"]=router_logits.is_contiguous()
        #             inter_data["router_logits_stride"]=router_logits.stride()
        #             inter_data["module_w13_weight"]= module.w13_weight.detach().cpu().clone()
        #             inter_data["module_w2_weight"]= module.w2_weight.detach().cpu().clone()
        #             inter_data["module_w13_weight_contiguous"]= module.w13_weight.is_contiguous()
        #             inter_data["module_w2_weight_contiguous"]= module.w2_weight.is_contiguous()
        #             inter_data["module_w13_weight_stride"]= module.w13_weight.stride()
        #             inter_data["module_w2_weight_stride"]= module.w2_weight.stride()
        #             if hasattr(module, "w13_bias"):
        #                 inter_data["module_w13_bias"]= module.w13_bias
        #                 inter_data["module_w13_bias_contiguous"]= module.w13_bias.is_contiguous()
        #                 inter_data["module_w13_bias_stride"]= module.w13_bias.stride()
        #             if hasattr(module, "w2_bias"):
        #                 inter_data["module_w2_bias"]= module.w2_bias
        #                 inter_data["module_w2_bias_contiguous"]= module.w2_bias.is_contiguous()
        #                 inter_data["module_w2_bias_stride"]= module.w2_bias.stride()
        #             logger.info("module:{}".format(module))

        #             # Matrix multiply.
        #             final_hidden_states = module.quant_method.apply(
        #                 layer=module,
        #                 x=hidden_states,
        #                 router_logits=router_logits,
        #                 top_k=module.top_k,
        #                 renormalize=module.renormalize,
        #                 use_grouped_topk=module.use_grouped_topk,
        #                 global_num_experts=module.global_num_experts,
        #                 expert_map=module.expert_map,
        #                 topk_group=module.topk_group,
        #                 num_expert_group=module.num_expert_group,
        #                 custom_routing_function=module.custom_routing_function,
        #                 scoring_func=module.scoring_func,
        #                 routed_scaling_factor=module.routed_scaling_factor,
        #                 e_score_correction_bias=module.e_score_correction_bias,
        #                 activation=module.activation,
        #                 apply_router_weight_on_input=module.apply_router_weight_on_input,
        #                 enable_eplb=module.enable_eplb,
        #                 expert_load_view=module.expert_load_view,
        #                 logical_to_physical_map=module.logical_to_physical_map,
        #                 logical_replica_count=module.logical_replica_count,
        #             )
        #             inter_data["final_hidden_states"]=final_hidden_states.detach().cpu().clone()
        #             inter_data["top_k"]=module.top_k
        #             inter_data["renormalize"]=module.renormalize
        #             inter_data["use_grouped_topk"]=module.use_grouped_topk
        #             inter_data["global_num_experts"]=module.global_num_experts
        #             inter_data["expert_map"]=module.expert_map
        #             inter_data["topk_group"]=module.topk_group
        #             inter_data["num_expert_group"]=module.num_expert_group
        #             inter_data["custom_routing_function"]=module.custom_routing_function
        #             inter_data["scoring_func"]=module.scoring_func
        #             inter_data["routed_scaling_factor"]=module.routed_scaling_factor
        #             inter_data["e_score_correction_bias"]=module.e_score_correction_bias
        #             inter_data["apply_router_weight_on_input"]=module.apply_router_weight_on_input
        #             inter_data["activation"]=module.activation
        #             inter_data["enable_eplb"]=module.enable_eplb
        #             inter_data["expert_load_view"]=module.expert_load_view
        #             inter_data["logical_to_physical_map"]=module.logical_to_physical_map
        #             inter_data["logical_replica_count"]=module.logical_replica_count
        #             worker_helper.moe_hook_data["inter_states"].append(inter_data)
        #             # You can dump intermediate variables here, for example:
        #             # torch.save({
        #             #     "hidden_states": hidden_states.detach().cpu(),
        #             #     "router_logits": router_logits.detach().cpu(),
        #             #     "final_hidden_states": final_hidden_states.detach().cpu() if isinstance(final_hidden_states, torch.Tensor) else None,
        #             # }, f"debug_moe_intermediate_{module_name}.pth")
                    
        #             if shared_output is not None:
        #                 assert not isinstance(final_hidden_states, tuple)
        #                 assert module.shared_experts is not None
        #                 final_hidden_states = (
        #                     shared_output,
        #                     final_hidden_states,
        #                 )
        #             elif module.zero_expert_num is not None and module.zero_expert_num > 0:
        #                 assert isinstance(final_hidden_states, tuple)
        #                 final_hidden_states, zero_expert_result = final_hidden_states
                    
        #             def reduce_output(
        #                 states: torch.Tensor, do_combine: bool = True
        #             ) -> torch.Tensor:
        #                 if do_naive_dispatch_combine and do_combine:
        #                     states = get_ep_group().combine(states, module.is_sequence_parallel)
                        
        #                 if (
        #                     not module.is_sequence_parallel
        #                     and module.reduce_results
        #                     and (module.tp_size > 1 or module.ep_size > 1)
        #                 ):
        #                     states = module.maybe_all_reduce_tensor_model_parallel(states)
                        
        #                 return states
                    
        #             if module.shared_experts is not None:
        #                 output = (
        #                     reduce_output(final_hidden_states[0], do_combine=False),
        #                     reduce_output(final_hidden_states[1]),
        #                 )
        #             elif module.zero_expert_num is not None and module.zero_expert_num > 0:
        #                 assert isinstance(final_hidden_states, torch.Tensor)
        #                 output = reduce_output(final_hidden_states) + zero_expert_result
        #             else:
        #                 output = reduce_output(final_hidden_states)
            
        #     # ========== End of forward_impl logic from layer.py ==========
            
        #     # Capture outputs
        #     if isinstance(output, torch.Tensor):
        #         output_data = output.detach().cpu().clone()
        #     elif isinstance(output, tuple):
        #         output_data = tuple(
        #             out.detach().cpu().clone() if isinstance(out, torch.Tensor) else out 
        #             for out in output
        #         )
        #     else:
        #         output_data = output
        #     worker_helper.moe_hook_data["outputs"].append(output_data)
            
        #     logger.info(f"MoE captured outputs - Module: {module_name}, "
        #                f"Output shape: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
        #     worker_helper.save_moe_hook_data()
            
        #     return output
        
        # ========== New implementation with forward_cuda logic ==========
        # quant_method = module.quant_method
        
        # def wrapped_forward_impl(hidden_states, router_logits):
        #     # Capture inputs
        #     input_data = {
        #         "hidden_states": hidden_states.detach().cpu().clone(),
        #         "router_logits": router_logits.detach().cpu().clone(),
        #     }
        #     worker_helper.moe_hook_data["inputs"].append(input_data)
            
        #     logger.info(f"MoE capturing inputs - Module: {module_name}, "
        #                f"Hidden states shape: {hidden_states.shape}, "
        #                f"Router logits shape: {router_logits.shape}")
            
        #     # ========== Start of forward_cuda logic from layer.py ==========
        #     # Get parameters from module/layer
        #     x = hidden_states
        #     layer = module
        #     use_grouped_topk = module.use_grouped_topk
        #     top_k = module.top_k
        #     renormalize = module.renormalize
        #     topk_group = module.topk_group
        #     num_expert_group = module.num_expert_group
        #     global_num_experts = module.global_num_experts
        #     expert_map = module.expert_map
        #     custom_routing_function = module.custom_routing_function
        #     scoring_func = module.scoring_func
        #     routed_scaling_factor = module.routed_scaling_factor
        #     e_score_correction_bias = module.e_score_correction_bias
        #     apply_router_weight_on_input = module.apply_router_weight_on_input
        #     activation = module.activation
        #     enable_eplb = module.enable_eplb
        #     expert_load_view = module.expert_load_view
        #     logical_to_physical_map = module.logical_to_physical_map
        #     logical_replica_count = module.logical_replica_count
            
        #     zero_expert_num = getattr(layer, "zero_expert_num", 0)
        #     zero_expert_type = getattr(layer, "zero_expert_type", None)
            
        #     # Import FusedMoE for select_experts
        #     from vllm.model_executor.layers.fused_moe import FusedMoE
            
        #     # Step 1: Select experts (this is where routing happens)
        #     topk_weights, topk_ids, zero_expert_result = FusedMoE.select_experts(
        #         hidden_states=x,
        #         router_logits=router_logits,
        #         use_grouped_topk=use_grouped_topk,
        #         top_k=top_k,
        #         renormalize=renormalize,
        #         topk_group=topk_group,
        #         num_expert_group=num_expert_group,
        #         custom_routing_function=custom_routing_function,
        #         scoring_func=scoring_func,
        #         routed_scaling_factor=routed_scaling_factor,
        #         e_score_correction_bias=e_score_correction_bias,
        #         indices_type=quant_method.topk_indices_dtype,
        #         enable_eplb=enable_eplb,
        #         expert_map=expert_map,
        #         expert_load_view=expert_load_view,
        #         logical_to_physical_map=logical_to_physical_map,
        #         logical_replica_count=logical_replica_count,
        #         global_num_experts=global_num_experts,
        #         zero_expert_num=zero_expert_num,
        #         zero_expert_type=zero_expert_type,
        #     )
            
        #     # ===== Dump intermediate variables after expert selection =====
        #     inter_data = {
        #         "hidden_states": x.detach().cpu().clone(),
        #         "router_logits": router_logits.detach().cpu().clone(),
        #         "topk_weights": topk_weights.detach().cpu().clone(),
        #         "topk_ids": topk_ids.detach().cpu().clone(),
        #         "zero_expert_result": zero_expert_result.detach().cpu().clone() if isinstance(zero_expert_result, torch.Tensor) else zero_expert_result,
        #         "hidden_states_contiguous": x.is_contiguous(),
        #         "hidden_states_stride": x.stride(),
        #         "router_logits_contiguous": router_logits.is_contiguous(),
        #         "router_logits_stride": router_logits.stride(),
        #         "module_w13_weight": layer.w13_weight.detach().cpu().clone(),
        #         "module_w2_weight": layer.w2_weight.detach().cpu().clone(),
        #         "module_w13_weight_contiguous": layer.w13_weight.is_contiguous(),
        #         "module_w2_weight_contiguous": layer.w2_weight.is_contiguous(),
        #         "module_w13_weight_stride": layer.w13_weight.stride(),
        #         "module_w2_weight_stride": layer.w2_weight.stride(),
        #         "top_k": top_k,
        #         "renormalize": renormalize,
        #         "use_grouped_topk": use_grouped_topk,
        #         "global_num_experts": global_num_experts,
        #         "expert_map": expert_map,
        #         "topk_group": topk_group,
        #         "num_expert_group": num_expert_group,
        #         "scoring_func": scoring_func,
        #         "routed_scaling_factor": routed_scaling_factor,
        #         "apply_router_weight_on_input": apply_router_weight_on_input,
        #         "activation": activation,
        #         "enable_eplb": enable_eplb,
        #     }
        #     if hasattr(layer, "w13_bias") and layer.w13_bias is not None:
        #         inter_data["module_w13_bias"] = layer.w13_bias.detach().cpu().clone()
        #         inter_data["module_w13_bias_contiguous"] = layer.w13_bias.is_contiguous()
        #         inter_data["module_w13_bias_stride"] = layer.w13_bias.stride()
        #     if hasattr(layer, "w2_bias") and layer.w2_bias is not None:
        #         inter_data["module_w2_bias"] = layer.w2_bias.detach().cpu().clone()
        #         inter_data["module_w2_bias_contiguous"] = layer.w2_bias.is_contiguous()
        #         inter_data["module_w2_bias_stride"] = layer.w2_bias.stride()
            
        #     logger.info(f"Module: {module_name}, topk_weights shape: {topk_weights.shape}, topk_ids shape: {topk_ids.shape}")
            
        #     # Step 2: Apply the actual MoE computation
        #     if quant_method.rocm_aiter_moe_enabled:
        #         assert quant_method.fused_experts is None
        #         result = quant_method.rocm_aiter_fused_experts(
        #             hidden_states=x,
        #             w1=layer.w13_weight,
        #             w2=layer.w2_weight,
        #             topk_weights=topk_weights,
        #             topk_ids=topk_ids,
        #             expert_map=expert_map,
        #             activation=activation,
        #             apply_router_weight_on_input=apply_router_weight_on_input,
        #             use_asm=quant_method.rocm_aiter_use_asm,
        #         )
        #     elif quant_method.flashinfer_cutlass_moe_enabled:
        #         result = quant_method.flashinfer_cutlass_moe(
        #             hidden_states=x,
        #             w1=layer.w13_weight,
        #             w2=layer.w2_weight,
        #             topk_weights=topk_weights,
        #             topk_ids=topk_ids,
        #             activation=activation,
        #             apply_router_weight_on_input=apply_router_weight_on_input,
        #         )
        #     elif quant_method.fused_experts is not None:
        #         if quant_method.moe.has_bias:
        #             raise ValueError("FusedMoEModularKernel does not support bias.")
        #         result = quant_method.fused_experts(
        #             hidden_states=x,
        #             w1=layer.w13_weight,
        #             w2=layer.w2_weight,
        #             topk_weights=topk_weights,
        #             topk_ids=topk_ids,
        #             inplace=True,
        #             activation=activation,
        #             apply_router_weight_on_input=apply_router_weight_on_input,
        #             global_num_experts=global_num_experts,
        #             expert_map=expert_map,
        #         )
        #     else:
        #         from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        #         assert fused_experts is not None
        #         result = fused_experts(
        #             hidden_states=x,
        #             w1=layer.w13_weight,
        #             w2=layer.w2_weight,
        #             topk_weights=topk_weights,
        #             topk_ids=topk_ids,
        #             inplace=True,
        #             activation=activation,
        #             quant_config=quant_method.moe_quant_config,
        #             apply_router_weight_on_input=apply_router_weight_on_input,
        #             global_num_experts=global_num_experts,
        #             expert_map=expert_map,
        #         )
            
        #     # Step 3: Handle zero experts if needed
        #     if zero_expert_num != 0 and zero_expert_type is not None:
        #         assert not isinstance(result, tuple), (
        #             "Shared + zero experts are mutually exclusive not yet supported"
        #         )
        #         output = (result, zero_expert_result)
        #     else:
        #         output = result
            
        #     # ===== Dump final result =====
        #     if isinstance(output, tuple):
        #         inter_data["result"] = output[0].detach().cpu().clone()
        #         inter_data["zero_expert_result"] = output[1].detach().cpu().clone() if len(output) > 1 else None
        #     else:
        #         inter_data["result"] = output.detach().cpu().clone()
            
        #     worker_helper.moe_hook_data["inter_states"].append(inter_data)
            
        #     # ========== End of forward_cuda logic ==========
            
        #     # Capture outputs
        #     if isinstance(output, torch.Tensor):
        #         output_data = output.detach().cpu().clone()
        #     elif isinstance(output, tuple):
        #         output_data = tuple(
        #             out.detach().cpu().clone() if isinstance(out, torch.Tensor) else out 
        #             for out in output
        #         )
        #     else:
        #         output_data = output
        #     worker_helper.moe_hook_data["outputs"].append(output_data)
            
        #     logger.info(f"MoE captured outputs - Module: {module_name}, "
        #                f"Output shape: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
        #     worker_helper.save_moe_hook_data()
            
        #     return output
        # # ========== End of new implementation ==========
        
        # return wrapped_forward_impl
    
    # def save_moe_hook_data(self, save_path=None):
    #     """Save captured MoE inputs and outputs to a file."""
    #     if save_path is None:
    #         rank = dist.get_rank() if dist.is_initialized() else 0
    #         save_path = f"/apps/zhaobing/fused_moe_have_shuffle_rank_{rank}.pth"
        
    #     torch.save(self.moe_hook_data, save_path)
    #     logger.info(f"Saved MoE hook data to {save_path}, captured {len(self.moe_hook_data['inputs'])} forward passes")
    #     return save_path

    def reload_model(self):
        if not self.weight_loaded:
            self.wake_up(["weights"])
            self.weight_loaded = True

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # before updating the parameters, we need to reinitialize the previously released model
        self.reload_model()
        patch_vllm_moe_model_weight_loader(self.model_runner.model)
        self.model_runner.model.load_weights(weights=weights)

    def load_states(self):
        self.reload_model()
        if not self.kv_cache_loaded:
            self.wake_up(["kv_cache"])
            self.kv_cache_loaded = True
        if vllm.__version__ < "0.8.5" and  self.buffers is not None:
            # https://github.com/vllm-project/vllm/issues/16564
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self.buffers:
                    buffer.data.copy_(self.buffers[name].data)
            self.buffers = None
        if not self.have_shuffled_weights:
            logger.info("current_platform:{}".format(current_platform))
            if current_platform.is_rocm():
                device_config = self.device_config
                load_config = self.vllm_config.load_config
                load_device = (
                    device_config.device if load_config.device is None else load_config.device
                )
                logger.info("device_config: {}".format(device_config))
                logger.info("load_config: {}".format(load_config))
                target_device = torch.device(load_device)
                with set_default_torch_dtype(self.model_config.dtype):
                    process_weights_after_loading(self.model_runner.model, self.model_config, target_device)
                # for _, module in self.model_runner.model.named_modules():
                #     if "model.layers.0.mlp.experts" in _ and isinstance(module, FusedMoE):
                #         logger.info("module_name:{}".format(_))
                        
                #         # Wrap forward_impl method to capture inputs and outputs
                #         # This is more reliable than forward hooks for CustomOp classes
                #         if hasattr(module, 'forward_impl'):
                #             module.forward_impl = self._create_forward_impl_wrapper(module, _)
                #             logger.info(f"Wrapped forward_impl for {_}")
                #         else:
                #             logger.warning(f"Module {_} does not have forward_impl method, skipping")
                                        
                # # # # # torch.cuda.synchronize()
                self.have_shuffled_weights = True
                logger.info("shuffled weights done")
        # torch.save(dict(self.model_runner.model.named_buffers()), f"/apps/zhaobing/use_aiter_have_shuffle_buffers_rank_{dist.get_rank()}.pth")
        # torch.save(self.model_runner.model.state_dict(), f"/apps/zhaobing/use_aiter_have_shuffle_state_dict_rank_{dist.get_rank()}.pth")

    def offload_states(self, level):
        assert (self.weight_loaded and self.kv_cache_loaded) or (not self.weight_loaded and not self.kv_cache_loaded)
        if not self.weight_loaded:
            return
        if vllm.__version__ < "0.8.5" and level == 2:
            # https://github.com/vllm-project/vllm/issues/16564
            model = self.model_runner.model
            self.buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}
        self.sleep(level)
        self.weight_loaded = False
        self.kv_cache_loaded = False
        if hasattr(self, 'recv_manager'):
            self.recv_manager.clear()
        gc.collect()
        current_platform.empty_cache()

    def setup_collective_group(self, comm_plan, backend, rank_in_cluster):
        self.model_update_comm_plan = getattr(self, "model_update_comm_plan", {})
        rank, comm_plan_args = get_dist_info_from_comm_plan(comm_plan, rank_in_cluster=rank_in_cluster,
                                                            rank_in_worker=dist.get_rank())
        if rank is None:
            logger.info(f"no comm_plan found for rank {rank_in_cluster}/{dist.get_rank()}")
            return
        group_name = comm_plan_args["group_name"]
        master_addr = comm_plan_args["master_addr"]
        master_port = comm_plan_args["master_port"]
        world_size = len(comm_plan_args["tgt_devices"]) + 1
        src_pp_rank = comm_plan_args["src_pp_rank"]
        collective.init_collective_group(world_size, rank, backend=backend, group_name=group_name,
                                         master_addr=master_addr, master_port=master_port)
        # A small all_reduce for warmup.
        collective.allreduce(torch.zeros(1).to(current_platform.device_type), group_name=group_name)
        self.model_update_comm_plan[src_pp_rank] = dict(rank=rank,
                                                        world_size=world_size,
                                                        src_pp_rank=src_pp_rank,
                                                        group_name=group_name,
                                                        comm_plan=comm_plan,
                                                        comm_plan_args=comm_plan_args)
        logger.info(f"warmup setup_collective_group: {group_name} rank: {rank} world_size: {world_size}")

    def broadcast_bucket(self, src_pp_rank, meta_infos, bucket_size):
        if src_pp_rank not in self.model_update_comm_plan:
            return
        comm_plan = self.model_update_comm_plan[src_pp_rank]
        buffer = torch.empty(bucket_size, dtype=torch.int8, device=current_platform.device_type)
        collective.broadcast(tensor=buffer, src_rank=0, group_name=comm_plan["group_name"])
        WorkerHelper.update_parameter_in_bucket(self, meta_infos, buffer, [dist.get_rank()])

    def broadcast_parameter(self, src_pp_rank, dtype, shape, parameter_name, is_lora=False):
        if src_pp_rank not in self.model_update_comm_plan:
            return
        comm_plan = self.model_update_comm_plan[src_pp_rank]
        weight = torch.empty(shape, dtype=dtype, device=current_platform.device_type)
        collective.broadcast(tensor=weight, src_rank=0, group_name=comm_plan["group_name"])
        WorkerHelper.update_parameter(self, parameter_name, weight, [dist.get_rank()], is_lora=is_lora)

    def update_parameter(self, parameter_name, weight, ranks_in_worker, is_lora=False):
        if is_lora:
            self.lora_params[parameter_name] = weight
            return
        if dist.get_rank() not in ranks_in_worker:
            return
        self.load_weights([(parameter_name, weight)])
        del weight

    def update_parameter_in_bucket(self, meta_infos, buffer, ranks_in_worker):
        if dist.get_rank() not in ranks_in_worker:
            return
        self.recv_manager = getattr(self, "recv_manager", RecvBucketManager())
        named_params = self.recv_manager.process_bucket(meta_infos, buffer)
        del buffer
        self.load_weights([(name, weight) for name, weight in named_params.items()])
        self.have_shuffled_weights = False