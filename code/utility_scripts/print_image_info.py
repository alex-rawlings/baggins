import argparse
import baggins as bgs


parser = argparse.ArgumentParser(description="Print imge metadata")
parser.add_argument(dest="img", help="image file", type=str)
args = parser.parse_args()


print(f"Metadata for image {args.img}: ")
print("--------")
bgs.plotting.get_meta(args.img)
