import os
import boto3
import logging
from pathlib import Path
from typing import Union


logger = logging.getLogger(__name__)


class DataProvider:

    def make_containers(self, path: Union[str, Path]) -> None:
        """
        Creates all non-existing parent folders or S3 Buckets in the path.
        
        Args:
            path: a path to a file, folder or to a S3 bucket.

        Returns:
            None
        """
        pass


    def list_objects(self, path: Union[str, Path]) -> list[str]:
        """
        List all objects (files and folders) in a folder or S3 bucket.
        
        Args:
            path: a path to a folder or to a S3 bucket.

        Returns:
            None
        """
        pass


    def get_object(self, path: Union[str, Path]) -> str:
        """
        Returns file content from a local file system or from a S3 bucket.
        
        Args:
            path: a path to a file in a local file system or in a S3 bucket.
        
        Returns:
            the file content.
        """
        pass

        
    def put_object(self, path: Union[str, Path], data: str) -> None:
        """
        Stores the data to a file in a local file system or in a S3 bucket.

        Args:
            path: a path to a file in a local file system or in a S3 bucket.
            data: the file content.

        Returns:
            None
        """
        pass


class LocalDataProvider(DataProvider):

    def __init__(self):
        super().__init__()


    def make_containers(self, path: Union[str, Path]) -> None:
        if isinstance(path, str):
            os.makedirs(path, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)


    def list_objects(self, path: Union[str, Path]) -> list[str]:
        return os.listdir(path)
    

    def get_object(self, path: Union[str, Path]) -> str:
        
        try:        
            with open(path, mode="r", encoding="utf-8") as fp:
                data = fp.read()
                return data
        except Exception as ex:
            logger.info(f"Reading file from path '{path}' failed: {ex}")


    def put_object(self, path: Union[str, Path], data: str) -> None:

        try:
            if isinstance(path, str):
                with open(path, mode="w", encoding="utf-8") as fp:
                    fp.write(data)
            else:
                path.write_text(data, encoding="utf-8")
        except Exception as ex:
            logger.info(f"Writting data to file at path '{path}' failed: {ex}")


class S3DataProvider(DataProvider):

    def __init__(self):
        super().__init__()
        self.client = boto3.client("s3")


    def _normalize_path(self, path: Union[str, Path], is_list_objects=False) -> tuple[str, str]:
        # In case when the path is created from the pathlib.Path object
        # the additional "/" will be remove from "s3://" prefix, so
        # e.g. AWS S3 path s3://bucket1 will become s3:/bucket after been
        # parsed by the pathlib.Path class.
        if isinstance(path, str):
            normalized_path = path.removeprefix("s3:/").removeprefix("/")
        else:
            normalized_path = "/".join(path.parts[1:])
        normalized_path = normalized_path.replace("\\", "/")
        if "/" in normalized_path:
            bucket, prefix = normalized_path.split("/", 1)
            # Ensure prefix ends with / for directory-like listing
            if is_list_objects and not prefix.endswith("/"):
                prefix += "/"
        else:
            bucket = normalized_path
            prefix = ""
        
        return bucket, prefix


    def make_containers(self, path: Union[str, Path]) -> None:
        pass


    def list_objects(self, path: Union[str, Path]) -> list[str]:
        try:
            bucket, prefix = self._normalize_path(path, is_list_objects=True)
            kwargs = dict(Bucket=bucket)
            if prefix:
                kwargs["Prefix"] = prefix
                kwargs["Delimiter"] = "/"
            response = self.client.list_objects(**kwargs)
            keys = []
            if "Contents" in response:
                # Extract just the file name from the file path.
                keys.extend([o["Key"].split("/")[-1] for o in response["Contents"]])
            return keys
        except Exception as ex:
            logger.info(f"Listing S3 objects failed: {ex}")


    def get_object(self, path: Union[str, Path]) -> str:
        try:
            bucket, key = self._normalize_path(path)
            response = self.client.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read()
            return content
        except Exception as ex:
            logger.info(f"Failed to get content of the object with the key '{key}' in the bucket '{path}': {ex}")
        

    def put_object(self, path: Union[str, Path], data: bytes) -> None:
        try:
            bucket, key = self._normalize_path(path)
            self.client.put_object(ACL="private", Body=data, Bucket=bucket, Key=key)
        except Exception as ex:
            logger.info(f"Failed to write data to the object with key '{key}' in the bucket '{path}': {ex}")


def get_data_provider(path: str) -> DataProvider:
   """
   Returns the local or the S3 data provider depending on the path prefix.

   Args:
       path: either local file system or S3 bucket location.

   Returns:
       a specialized data provider object.
   """
   return S3DataProvider() if path.startswith("s3://") else LocalDataProvider()

