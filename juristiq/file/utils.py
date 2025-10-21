import os
from io import StringIO
from pathlib import Path
from juristiq.config.consts import TEXT_ENCODING


class FileWriter:
    """Utility class for writting text to files."""
    def __init__(self, 
                 file_name: str,
                 output_folder_path: str, 
                 max_file_size: int):
        """
        Initializes new instance of the FileWriter class.

        Args:
            file_name: the initial name of the output file.
            output_folder_path: the path the output folder.
            max_file_size: the maximum file size in megabytes, if exceeded a new file is created.
        """

        self.output_folder_path = Path(output_folder_path)
        self.output_file_path = self.output_folder_path / os.path.basename(file_name)
        self.output_file_ext = os.path.splitext(file_name)[1]
        self.max_file_size = max_file_size * 1024 * 1024
        self.buffer = StringIO()
        self.current_buffer_size = 0

        self._prepare_output_folder()


    def __enter__(self):
        return self
    

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exits the runtime context related to this object and flushes the buffer content
        if the cursor is not at the beginning of the stream.
        """
        if self.buffer.tell():
            self._flush_buffer()


    def _prepare_output_folder(self) -> None:
        """
        Removes any files with the specific extension from the output folder.
        """
        file_ext = os.path.splitext(self.output_file_path)[1]

        if self.output_folder_path.exists():
            _ = [p.unlink() for p in self.output_folder_path.rglob(f"*{file_ext}")]
        else:
            self.output_folder_path.mkdir(parents=True, exist_ok=True)


    def _get_unique_path(self) -> Path:
        """
        Generate a unique file path by appending an index if the file already exists.

        Args:
            path (Path): The original file path.
            file_stem (str): The stem of the file name to be used for indexing.

        Returns:
            a path object representing a unique file path.
        """
        path = self.output_file_path

        if path.exists():
            file_index = len(list(self.output_folder_path.rglob(f"*{self.output_file_ext}"))) + 1
            path = self.output_folder_path / f"{path.stem}{file_index}{self.output_file_ext}"

        return path


    def _flush_buffer(self) -> None:
        """
        Write the contents of the buffer to a uniquely named file.

        Args:
            file_path (Path): The base file path where the buffer will be written.
        """
        output_path = self._get_unique_path()
        output_path.write_text(self.buffer.getvalue(), encoding="utf8")

        self._reset_buffer()


    def _reset_buffer(self) -> None:
        """
        Clears the buffer and resets the stream position.
        """
        self.buffer.truncate(0)
        self.buffer.seek(0)
        self.current_buffer_size = 0


    def write(self, text: str) -> None:
        """
        Write the text to the buffer if the buffer size exceeds the maximum size
        the buffer's content is written to the file.

        Args:
            text: a text to write.
        """
        bytes_count = len(text.encode(encoding=TEXT_ENCODING))

        if self.current_buffer_size + bytes_count > self.max_file_size:
            self._flush_buffer()
            self._reset_buffer()

        self.buffer.write(text)
        self.current_buffer_size += bytes_count

        

