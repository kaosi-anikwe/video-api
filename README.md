# Text to Video API with Midjourney and Stable Video Diffusion (SVD)

## Overview

Text to Video API using Midjourney and stable diffusion. The API uses [this](https://www.thenextleg.io/) Midjourney API and the svd model from Stability AI.

## Setup

Copy and paste contents of [commands.txt](commands.txt) into the docker command input box while creating the RunPod instance. Add port 5000 to the HTTP ports to be exposed.
Wait for the build to complete to get the `BASE_URL`

## Endpoints

### `POST /img2vid`

#### Description

Endpoint for staging image generation task.

#### Parameters (as `JSON`)

- `prompt` : The prompt that will be used to generate the image for animation.
- `image_url` : The link to the image to be used to generate the video.

`Note:` Only one of these should be submitted. The default is `prompt`.

#### Optional parameters

- `H` (Number): Height of the image (e.g., 576)
- `W` (Number): Width of the image (e.g., 1024)
- `T` (Number): Total number of frames generated (e.g., 24)
- `cond_aug` (Number, Range: 0.00 to 1.00): Conditioning augmentation (e.g., 0.02)
- `seed` (Number, Range: 0 to 1e9): Seed for randomization (e.g., 23)
- `decoding_t` (Number, Range: 1 to 1e9): Decode t frames at a time (e.g., 14)
- `fps` (Number, Minimum: 1): Frames per second (e.g., 6)
- `motion_bucket_id` (Number, Range: 0 to 512): How much movement you want (e.g., 127)
- `num_cols` (Number, Range: 1 to 10): Number of columns (e.g., 1)
- `steps` (Number, Range: 1 to 1000): Number of steps (e.g., 40)

#### Sampler

- `sampler` (String, Options: EulerEDMSampler, HeunEDMSampler, EulerAncestralSampler, DPMPP2SAncestralSampler, DPMPP2MSampler, LinearMultistepSampler): Type of sampler (e.g., EulerEDMSampler)

  - If `sampler` is EulerEDMSampler or HeunEDMSampler:

    - `s_churn` (Number, Default: 0.0): Churn parameter (e.g., 0.0)
    - `s_tmin` (Number, Default: 0.0): Minimum time (e.g., 0.0)
    - `s_tmax` (Number, Default: 999.0): Maximum time (e.g., 999.0)
    - `s_noise` (Number, Default: 1.0): Noise parameter (e.g., 1.0)

  - If `sampler` is EulerAncestralSampler or DPMPP2SAncestralSampler:

    - `s_noise` (Number, Range: 0.0 to 1.0, Default: 1.0): Noise parameter (e.g., 1.0)
    - `eta` (Number, Range: 0.0 to 1.0, Default: 1.0): Eta parameter (e.g., 1.0)

  - If `sampler` is LinearMultistepSampler:
    - `order` (Number, Minimum: 1, Default: 4): Order parameter (e.g., 4)

#### Discretization

- `discretization` (String, Options: EDMDiscretization, LegacyDDPMDiscretization): Type of discretization (e.g., EDMDiscretization)

  - If `discretization` is EDMDiscretization:
    - `sigma_min` (Number, Default: 0.0292): Minimum sigma value (e.g., 0.0292)
    - `sigma_max` (Number, Default: 14.6146): Maximum sigma value (e.g., 14.6146)
    - `rho` (Number, Default: 3.0): Rho parameter (e.g., 3.0)

#### Guider

- `guider` (String, Options: LinearPredictionGuider, VanillaCFG, IdentityGuider): Type of guider (e.g., LinearPredictionGuider)

  - If `guider` is VanillaCFG:

    - `cfg_scale` (Number, Minimum: 0.0, Default: 5.0): CFG scale parameter (e.g., 5.0)

  - If `guider` is LinearPredictionGuider:
    - `max_cfg_scale` (Number, Minimum: 1.0, Default: 1.5): Maximum CFG scale (e.g., 1.5)
    - `min_guidance_scale` (Number, Range: 1.0 to 10.0, Default: 1.0): Minimum guidance scale (e.g., 1.0)

`Note:` The request should be sent as form data and not JSON

Endpoint returns JSON object with task status.

### `GET /status`

#### Description

This endpoint retrieves tasks from the system. It returns either a JSON list of tasks or a JSON object, depending on whether the optional request argument "id" is specified.

#### Parameters

- `id` (Optional, String): The unique identifier for a specific task. If provided, the endpoint will return a JSON object for the specified task. If not provided, the endpoint will return a JSON list of all tasks.
