import gzip
from io import BytesIO

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Note: fastparquet handles Timedelta, pyarrow doesn't
def write_s3_parquet(df, bucket, file_path, engine="fastparquet"):
    url = f"s3://{bucket}/{file_path}.parquet"
    df.to_parquet(url, engine=engine)


def read_s3_parquet(bucket, file_path, engine="fastparquet"):
    url = f"s3://{bucket}/{file_path}.parquet"
    df = pd.read_parquet(url, engine=engine)
    return df


# Note: to_parquet to a GzipFile object does not work with fastparquet engine
def write_s3_parquet_gzip(df, bucket, file_path, preserve_index=False):
    s3_resource = boto3.resource("s3")
    gz_buffer = BytesIO()
    with gzip.GzipFile(mode="w", fileobj=gz_buffer) as gz_file:
        # this trick is there instead of the simpler:
        # df.to_parquet(gz_file, index=preserve_index, compression='gzip', engine="pyarrow")
        # for the case where columns are a multiindex
        table = pa.Table.from_pandas(df, preserve_index=preserve_index)
        pq.write_table(table, gz_file, compression="gzip")
    s3_object = s3_resource.Object(bucket, file_path + ".parquet.gzip")
    s3_object.put(Body=gz_buffer.getvalue())
    gz_buffer.close()


def read_s3_parquet_gzip(bucket, file_path):
    s3_resource = boto3.resource("s3")
    gzipfile = s3_resource.Object(bucket, file_path + ".parquet.gzip")
    gzipfile = gzipfile.get()["Body"].read()
    gzipfile = BytesIO(gzipfile)
    with gzip.GzipFile(fileobj=gzipfile) as gzipfile:
        df = pd.read_parquet(gzipfile, engine="pyarrow")
    # if df has columns supposed to be arranged in a multiindex,
    # use this to reconstruct the MultiIndex:
    # df.columns = pd.MultiIndex.from_tuples(
    #     [tuple(st[1:-1] for st in label[1:-1].split(', ')) for label in df.columns]
    #     )
    return df
