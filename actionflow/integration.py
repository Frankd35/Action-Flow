import torch
import types
from .modeling.pipeline import ActionFlowPipeline

def enable_actionflow(vla_model, max_new_tokens=7):
    """
    Enables ActionFlow acceleration on an existing OpenVLA model instance.
    This performs in-place monkey-patching of the model's prediction methods.
    
    Args:
        vla_model: The Prismatic/OpenVLA model instance.
        max_new_tokens: The depth of the pipeline (K).
    """
    print(f"[ActionFlow] Initializing pipeline with depth K={max_new_tokens}...")
    
    # 1. Initialize the Pipeline Engine
    # This wraps the underlying LLM but shares weights
    pipeline_engine = ActionFlowPipeline(
        vla_model.language_model, 
        max_token=max_new_tokens
    )
    
    # Attach engine to model to prevent garbage collection
    vla_model.actionflow_engine = pipeline_engine
    
    # 2. Define the new accelerated predict method
    def predict_action_accelerated(self, input_ids=None, pixel_values=None, unnorm_key=None, **kwargs):
        """
        Accelerated action prediction using ActionFlow.
        """
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
            
        # A. Vision + Text Embedding (Standard VLA frontend)
        # We manually run the frontend to get the fused embeddings
        multimodal_embeddings = self._get_multimodal_embeddings(input_ids, pixel_values)
        
        # B. Initialize Resources (Lazy init based on input shape)
        # We check the shape of the incoming prefill
        prefill_len = multimodal_embeddings.shape[1]
        self.actionflow_engine.init_resources(prefill_len=prefill_len, max_new_tokens=max_new_tokens)
        
        # C. Run Pipeline
        # Note: This executes one step of the macro-pipeline.
        # It takes the current frame's embeddings and returns the finished action
        # from (K-1) frames ago.
        output_ids = self.actionflow_engine.pipe_forward(multimodal_embeddings)
        
        # D. Decode Action (Standard VLA backend)
        # Convert IDs back to continuous actions
        action_dim = self.get_action_dim(unnorm_key)
        
        # Slice the last `action_dim` tokens to ensure we get the action part
        # Logic matches OpenVLA: vocab_size - token_id
        predicted_ids = output_ids[0, -action_dim:].cpu().numpy()
        
        discretized_actions = self.vocab_size - predicted_ids
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )
        normalized_actions = self.bin_centers[discretized_actions]
        
        # Un-normalize
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high = np.array(action_norm_stats["q99"])
        action_low = np.array(action_norm_stats["q01"])
        
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions

    # 3. Helper for Multimodal Embeddings
    def _get_multimodal_embeddings(self, input_ids, pixel_values):
        # Vision Backbone
        patch_features = self.vision_backbone(pixel_values)
        # Projector
        projected_patches = self.projector(patch_features)
        # Text Embeddings
        input_embeddings = self.get_input_embeddings()(input_ids)
        
        # Concatenate: <BOS> + Image + Instruction
        # Assuming standard OpenVLA formatting
        embeddings = torch.cat(
            [input_embeddings[:, :1, :], projected_patches, input_embeddings[:, 1:, :]], 
            dim=1
        )
        return embeddings

    # 4. Apply Patches
    import numpy as np # Need numpy inside for the closure
    
    # Bind methods
    vla_model._get_multimodal_embeddings = types.MethodType(_get_multimodal_embeddings, vla_model)
    vla_model.predict_action = types.MethodType(predict_action_accelerated, vla_model)
    
    print("[ActionFlow] Model successfully patched! `predict_action` is now accelerated.")
    return vla_model