# Picasso Protocol v1
this is official einstein-picasso-odyssei project repo


**How to use it?**

In toWebPage dir, there are index.html and server.py. Execute server.py and wait for loading models(Caution: You need pre-trained model). Then double click the index.html. Have Fun!



**Model Data**

https://drive.google.com/drive/folders/1p2EyQxCJMCiGHDhB0LjuIHfLjvf5jvvE?usp=sharing



**Referance**

RFNNS: Robust Fixed Neural Network Steganography with Popular Deep Generative Models(https://arxiv.org/pdf/2505.04116)

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(https://arxiv.org/abs/1810.04805)



**DataSet**

https://huggingface.co/datasets/Salesforce/wikitext/viewer/wikitext-2-raw-v1




**Needed**

Python 3.11+

PyTorch 2.7.1

Transformers 4.54.0

Datasets 4.0.0

Pillow 11.2.1

NumPy 2.3.1

Matplotlib 3.10.3

Tkinter


## Compact Public Encoder Prototype

The current compact prototype separates the model into asymmetric roles:

- `Artist Y`: public encoder. Anyone can convert text into a latent representation.
- `Artist X`: private decoder. Only the receiver keeps this model and restores text.

This is a prototype inspired by asymmetric encryption. It is not a cryptographic
security guarantee yet.

The WikiText-trained model accepts up to 159 UTF-8 bytes and creates a latent
sequence with shape `[1, 160, 128]`.

### Split the trained bundle

```powershell
conda run -n picasso-gpu python ver.3\train\split_public_private_models.py
```

Generated files:

```text
models/public/artist_y_public_encoder.pt
models/private/artist_x_private_decoder.pt
```

Share only the `models/public/` directory. Never distribute:

```text
models/private/
ver.3/train/wikitext_checkpoints_final/
```

The training directory contains a development bundle that can recreate the
private decoder.

### Easy CLI

Check the configured models:

```powershell
.\picasso.cmd status
.\picasso.cmd verify-public
```

Create a latent JSON file with the public Artist Y encoder:

```powershell
.\picasso.cmd encode "My Secret"
```

Restore the text with the private Artist X decoder:

```powershell
.\picasso.cmd decode picasso_latent.json
```

Custom output name:

```powershell
.\picasso.cmd encode "My Secret" -o message.json
```

The equivalent long form is `conda run -n picasso-gpu python picasso.py ...`.

### Separate Role CLIs

For presentations and deployment, use the separated commands:

```powershell
.\artist-y-public.cmd status
.\artist-y-public.cmd encode "My Secret" -o captured_latent.json

.\artist-x-private.cmd status
.\artist-x-private.cmd decode captured_latent.json
```

Run the local attacker demonstration:

```powershell
.\attacker-demo.cmd captured_latent.json --known-text "My Secret"
```

The full recording sequence is documented in [DEMO_SCRIPT_KO.md](./DEMO_SCRIPT_KO.md).

Run the decoder inversion vulnerability demonstration:

```powershell
.\attacker-inversion-demo.cmd --sizes 100 500 1000 --epochs 100
```

Measured results and analysis are documented in [ATTACK_REPORT_KO.md](./ATTACK_REPORT_KO.md).

Run the stochastic latent shielding demo:

```powershell
.\artist-y-shielded.cmd "My Secret" -o shielded_latent.json
.\artist-x-shielded.cmd shielded_latent.json
.\attacker-inversion-demo.cmd --public-model models\public_v4\artist_y_public_encoder.pt --sizes 100 500 1000 --epochs 60 --test-size 400 --public-dropout 0.65 --public-noise 1.0 --output attack_results\decoder_inversion_v4_dropout65_noise1_results.json
```

The V4 shielded model uses latent dropout/noise learned by the private decoder.
It does not solve the security problem completely, but it raises the inversion
attack cost without adding an external cryptographic algorithm. In the measured
demo, exact-match inversion dropped from `0.9875` to `0.9000` at 1000 chosen
plaintext pairs, while normal decoding stayed around `0.976` exact match.

### V5 Split Brain Fragment

V5 trains one `PicassoBrain` and splits it only at export time:

```text
Artist Y public brain fragment -> ambiguous latent
Artist X private brain fragment -> text reconstruction
```

Run:

```powershell
.\artist-y-brain-v5.cmd encode "My Secret" -o brain_v5_latent.json
.\artist-x-brain-v5.cmd decode brain_v5_latent.json
```

Measured V5b results:

```text
normal exact match: 0.9577
1000-message encode+decode speed: ~0.138 ms/message
1000-pair inversion exact match: 0.7525
```

This is the strongest current neural-only defense in the repo. It does not make
the system cryptographically secure, but it makes surrogate decoder training
much less stable than the original public-encoder design.

### V6 UltraDrop Strong Mode

V6 increases the split-brain latent width and masks more of the public latent at
runtime. This trades some normal exact-match accuracy for a stronger inversion
attack reduction.

```powershell
.\artist-y-brain-v6.cmd encode "My Secret" -o brain_v6_latent.json
.\artist-x-brain-v6.cmd decode brain_v6_latent.json
```

Measured V6 UltraDrop results:

```text
normal validation exact match: 0.9415
1000-message encode+decode speed: ~0.140 ms/message
1000-pair inversion exact match: 0.6750
500-pair inversion exact match: 0.4725
```

Use V5b when you want higher normal decoding reliability. Use V6 when the demo
emphasizes attack resistance.

### Legacy Latent CLI

```powershell
conda run -n picasso-gpu python encode_latent.py --text "My Secret" --output latent_demo.json
conda run -n picasso-gpu python decode_latent.py --input latent_demo.json
```

### Latent API

Run:

```powershell
.\server-public.cmd
```

By default the server exposes only public encoding. It does not load the private
Artist X decoder. The receiver can enable local decoding explicitly:

```powershell
.\server-private.cmd
```

Endpoints:

```text
GET  /health
POST /encode-latent
POST /decode-latent
```

PNG encoding is intentionally left for the next implementation step.

