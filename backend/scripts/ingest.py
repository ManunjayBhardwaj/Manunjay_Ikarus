import argparse
import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'backend'))

from app.routes.ingest import ingest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=os.getenv('DATASET_PATH'))
    parser.add_argument('--backend', choices=['pinecone','faiss'], default=os.getenv('VECTOR_BACKEND','faiss'))
    args = parser.parse_args()
    print('Running ingest (this will call the same logic as the API)...')
    res = ingest()
    print(res)

if __name__ == '__main__':
    main()
