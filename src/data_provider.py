import os
import boto3
import logging

logger = logging.getLogger(__name__)



class DataProvider:

    def list_objects(self, path: str) -> list[str]:
        pass

    def get_object(self, path: str, name: str) -> str:
        pass
        
    def put_object(self, path: str, name: str, data: str) -> None:
        pass


class LocalDataProvider(DataProvider):

    def __init__(self):
        super().__init__()


    def list_objects(self, path: str) -> list[str]:
        return os.listdir(path)
    

    def get_object(self, path: str, name: str) -> str:
        
        obj_path = os.path.join(path, name)

        try:        
            with open(obj_path, mode="r", encoding="utf-8") as fp:
                data = fp.read()
                return data
        except Exception as ex:
            logger.info(f"Reading file from path '{obj_path}' failed: {ex}")


    def put_object(self, path: str, name: str, data: str) -> None:

        obj_path = os.path.join(path, name)

        try:
            with open(obj_path, mode="w", encoding="utf-8") as fp:
                fp.write(data)
        except Exception as ex:
            logger.info(f"Writting data to file at path '{obj_path}' failed: {ex}")


class S3DataProvider(DataProvider):

    def __init__(self):
        super().__init__()
        self.client = boto3.client("s3")


    def _normalize_path(self, path: str, is_list_objects=False) -> tuple[str, str]:

        normalized_path = path.removeprefix("s3://")
        if "/" in normalized_path:
            bucket, prefix = normalized_path.split("/", 1)
            # Ensure prefix ends with / for directory-like listing
            if is_list_objects and not prefix.endswith("/"):
                prefix += "/"
        else:
            bucket = normalized_path
            prefix = ""
        
        return bucket, prefix


    def list_objects(self, path: str) -> list[str]:
        """
        List objects in an S3 path.
        
        Args:
            path: S3 path in format 's3://bucket/prefix' or 'bucket/prefix'
        
        Returns:
            List of object keys. Returns empty list if path doesn't exist or has no objects.
        """
        
        try:
            bucket, prefix = self._normalize_path(path, is_list_objects=True)
            response = self.client.list_objects(Bucket=bucket, Prefix=prefix, Delimiter="/")
            keys = []
            if "Contents" in response:
                keys.extend([o["Key"] for o in response["Contents"]])
            return keys
        except Exception as ex:
            logger.info("Listing S3 objects failed: {ex}")


    def get_object(self, path: str, name: str) -> str:

        try:
            bucket, _ = self._normalize_path(path)
            response = self.client.get_object(Bucket=bucket, Key=name)
            content = response["Body"].read()
            return content
        except Exception as ex:
            logger.info(f"Failed to get content of the object with the key '{name}' in the bucket '{path}': {ex}")
        

    def put_object(self, path: str, name: str, data: bytes) -> None:
        try:
            bucket, _ = self._normalize_path(path)
            self.client.put_object(ACL="private", Body=data, Bucket=bucket, Key=name)
        except Exception as ex:
            logger.info(f"Failed to write data to the object with key '{name}' in the bucket '{path}': {ex}")
