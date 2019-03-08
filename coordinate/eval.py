
def eval_one_batch(self, batch):
    enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
        get_input_from_batch(batch, use_cuda)
    dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
        get_output_from_batch(batch, use_cuda)

    encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
    s_t_1 = self.model.reduce_state(encoder_hidden)

    step_losses = []
    for di in range(min(max_dec_len, config.max_dec_steps)):
        y_t_1 = dec_batch[:, di]  # Teacher forcing
        final_dist, s_t_1, c_t_1,attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                    encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                    extra_zeros, enc_batch_extend_vocab, coverage, di)
        target = target_batch[:, di]
        gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
        step_loss = -torch.log(gold_probs + config.eps)
        if config.is_coverage:
            step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
            step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
            coverage = next_coverage

        step_mask = dec_padding_mask[:, di]
        step_loss = step_loss * step_mask
        step_losses.append(step_loss)

    sum_step_losses = torch.sum(torch.stack(step_losses, 1), 1)
    batch_avg_loss = sum_step_losses / dec_lens_var
    loss = torch.mean(batch_avg_loss)

    return loss.data[0]
