import argparse
import os
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MoCo Pre-Traind Model to DEiT')
    parser.add_argument('--input', default='', type=str, metavar='PATH', required=True,
                        help='path to moco pre-trained checkpoint')
    parser.add_argument('--output', default='', type=str, metavar='PATH', required=True,
                        help='path to output checkpoint in mae format')
    parser.add_argument('--grouped_bands', type=int, nargs='+', action='append',
                        default=[], help="Bands to group for moco views")
    args = parser.parse_args()
    print(args)

    if len(args.grouped_bands) == 0:
        args.grouped_bands = [[0, 6, 7, 8, 9], [1, 2, 3, 4, 5]]
        print(f"Grouping bands {args.grouped_bands}")

    # load input
    checkpoint = torch.load(args.input, map_location="cpu")
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
            # remove prefix
            weight = state_dict[k]

            # Hack to fix for split channel vit
            if k.endswith('patch_embed.proj.weight'):
                new_t = torch.zeros(weight.shape[0], 10, weight.shape[2], weight.shape[3])

                for i, (b1, b2) in enumerate(zip(*args.grouped_bands)):
                    new_t[:, b1] = weight[:, i]
                    new_t[:, b2] = weight[:, i]

                weight = new_t

            state_dict[k[len("module.base_encoder."):]] = weight
        # delete renamed or unused k
        del state_dict[k]

    # make output directory if necessary
    output_dir = os.path.dirname(args.output)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # save to output
    torch.save({'model': state_dict}, args.output)
