import torch.nn as nn
import torch
from transformers import AutoModel
from transformers.models.t5.modeling_t5 import T5LayerNorm

"""
encoder_outputs = self.encoder(
    input_ids=input_ids,
    attention_mask=attention_mask,
    inputs_embeds=inputs_embeds,
    head_mask=head_mask,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    return_dict=return_dict,
)
"""

"""
decoder_outputs = self.decoder(
    input_ids=decoder_input_ids,
    attention_mask=decoder_attention_mask,
    inputs_embeds=decoder_inputs_embeds,
    past_key_values=past_key_values,
    encoder_hidden_states=hidden_states,
    encoder_attention_mask=attention_mask,
    head_mask=decoder_head_mask,
    cross_attn_head_mask=cross_attn_head_mask,
    use_cache=use_cache,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    return_dict=return_dict,
)
"""

class BeIT_T5_Concepts(nn.Module):
  def __init__(self, vision_module, language_module, device = 'cpu', crossentropy_output=True):
    super(BeIT_T5_Concepts, self).__init__()

      # Load VIT
    self.image_encoder = AutoModel.from_pretrained(vision_module).to(device)

    # Load mT5
    language_model = T5ForConditionalGeneration.from_pretrained(language_module, from_flax = True).to(device)

    # T5LayerNorm
    self.layer_norm = T5LayerNorm(language_model.config.d_model).to(device)

    # T5 encoder
    self.concepts_encoder = language_model.encoder

    # T5 Decoder
    self.captions_decoder = language_model.decoder

    # T5 Head
    self.lm_head = language_model.lm_head

    self.padding_idx = self.captions_decoder.config.pad_token_id
    self.crossentropy_output = crossentropy_output

    self.update_model_config()

  def update_model_config(self):
      self.captions_decoder.config.decoder_start_token_id = 1

  def _shift_right(self, input_ids):
      decoder_start_token_id = self.captions_decoder.config.decoder_start_token_id
      pad_token_id = self.padding_idx

      if decoder_start_token_id is None:
          raise ValueError(
              "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. "
              "See T5 docs for more information."
          )

      shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id).to(device)
      shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)

      if pad_token_id is None:
          raise ValueError("self.model.config.pad_token_id has to be defined.")
      # replace possible -100 values in labels by `pad_token_id`
      shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

      return shifted_input_ids

  def forward(self, samples):

    device = self.image_encoder.device

    pixel_values = samples['pixel_values'].to(device)
    labels = samples['labels'].to(device)
    attention_mask = samples['attention_mask'].to(device)
    tokenized_label_name = samples['tokenized_label_name'].to(device)
    tokenized_label_mask = samples['tokenized_label_mask'].to(device)

    # Encoding Image
    image_features = self.image_encoder(pixel_values = pixel_values).last_hidden_state # batch_size, 193, 768
    image_features_ = self.layer_norm(image_features)

    # Encoding Concepts
    concept_features = self.concepts_encoder(
        input_ids = tokenized_label_name,
        attention_mask = tokenized_label_mask
    ).last_hidden_state # batch_size, num_concepts, 768

    # Concat Image and Concepts Features
    cross_features = self.layer_norm(torch.cat([image_features_, concept_features], dim = 1)) # batch_size, 193 + num_concepts, 768
    cross_mask = torch.cat([torch.ones(image_features_.size(0), image_features_.size(1)).to(device), tokenized_label_mask], dim = 1) # batch_size, 193 + num_concepts

    # Generate Answer
    labels_ = self._shift_right(labels)
    decoded_captions = self.captions_decoder(input_ids = labels_,
                        attention_mask = attention_mask,
                        encoder_hidden_states = cross_features,
                        encoder_attention_mask = cross_mask
                        ).last_hidden_state # batch_size, answer_length, 768

    output = self.lm_head(decoded_captions)

    # Roll the answer here and
    # labels = samples['answer_'].input_ids.roll(shifts=-1, dims=1)
    # labels[:, -1] = self.padding_idx

    # tranpose for calculate the crossentropy loss:
    if self.crossentropy_output:
      output_ = output.transpose(-1, 1)

    return (output_, labels)