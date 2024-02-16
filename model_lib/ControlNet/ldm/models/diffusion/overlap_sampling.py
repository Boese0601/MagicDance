import random
pred_all = torch.zeros_like(latents)
counts = torch.zeros(num_frames).cuda()
offset = random.randint(0, num_frames-1) # random offset
for start_idx in range(offset, offset+num_frames-16+1, 12): #??
    indices = torch.arange(start_idx, start_idx + 16) % num_frames
    pred_pos = pipeline(latents[:,:,indices], timestep, **cond).sample
    pred_neg = pipeline(latents[:,:,indices], timestep, **uncond).sample
    pred = pred_neg + guidance_weight * (pred_pos - pred_neg)
    pred_all[:,:,indices] += pred
    counts[indices] += 1
# print(timestep, counts)
pred = pred_all / counts.reshape(1,1,-1,1,1)
latents = scheduler.step(
    pred,
    args.unet_prediction,
    timestep,
    latents
).prev_sample