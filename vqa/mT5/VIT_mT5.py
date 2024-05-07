import torch.nn as nn
import torch
from transformers import AutoModel

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

class VIT_mT5(nn.Module):
  def __init__(self, vision_module = 'google/vit-base-patch16-224-in21k', language_module = 'google/mt5-base', device = 'cpu', crossentropy_output=True):
    super(VIT_mT5, self).__init__()

      # Load VIT
    self.image_encoder = AutoModel.from_pretrained(vision_module).to(device)

    # Load mT5
    language_model = AutoModel.from_pretrained(language_module).to(device)

    # mT5 encoder
    self.question_encoder = language_model.encoder

    # mT5 Decoder
    self.answer_decoder = language_model.decoder

    # Head
    embedding_size = language_model.config.d_model
    vocab_size = language_model.config.vocab_size

    self.classifier = nn.Linear(embedding_size, vocab_size)

    # Norm Cross Feature
    self.norm_input = nn.LayerNorm(embedding_size)

    self.padding_idx = self.question_encoder.config.pad_token_id
    self.crossentropy_output = crossentropy_output

  def forward(self, samples):

    device = self.image_encoder.device
    batch_size = samples['image_'].shape[0]

    samples['image_'] = samples['image_'].to(device)
    samples['question_'] = samples['question_'].to(device)
    samples['answer_'] = samples['answer_'].to(device)

    # Prepare Input Features
    image_features = self.image_encoder(samples['image_']).last_hidden_state # batch_size, 193, 768

    # Image Mask
    image_mask = torch.ones(batch_size, image_features.shape[1], device=device)

    # Embedding Question
    encoded_question = self.question_encoder(input_ids = samples["question_"].input_ids, 
                                            attention_mask = samples["question_"].attention_mask).last_hidden_state # batch_size, question_length, 768

    # Image + Question
    cross_mask = torch.cat([image_mask, samples['question_'].attention_mask], dim = 1)
    cross_feature = self.norm_input(torch.cat([image_features, encoded_question], dim = 1)) # batch_size, 193 + question_length, 768

    # Generate Answer
    output = self.answer_decoder(input_ids = samples['answer_'].input_ids, 
                        attention_mask = samples['answer_'].attention_mask,
                        encoder_hidden_states = cross_feature,
                        encoder_attention_mask = cross_mask
                        ).last_hidden_state # batch_size, answer_length, 768
    
    # Roll the answer here and
    labels = samples['answer_'].input_ids.roll(shifts=-1, dims=1)
    labels[:, -1] = self.padding_idx

    # tranpose for calculate the crossentropy loss:
    if self.crossentropy_output:
      output_ = output.transpose(-1, 1)

    return (output, output_, labels)