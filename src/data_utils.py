from typing import List, Any, Callable


class DataBatch:

    def __init__(self, batch_size: int, data_store_type: Callable=list):
        self.data = data_store_type([])
        self.batch_size = batch_size


    def add(self, item: Any | List[Any]) -> None:
        """
        Add an item or list of items to the data collection.
        
        Args:
            item: a single item or list of items to add.
        """
        if isinstance(self.data, set):
            self._add_to_set(item)
        else:
            self._add_to_sequence(item)


    def _add_to_set(self, item: Any | List[Any]) -> None:
        
        if isinstance(item, list):
            self.data.update(item)
        else:
            self.data.add(item)


    def _add_to_sequence(self, item: Any | List[Any]) -> None:
        
        if isinstance(item, list):
            self.data.extend(item)
        else:
            self.data.append(item)
        

    def is_full(self) -> bool:
        return len(self.data) >= self.batch_size
    

    def has_items(self) -> bool:
        return len(self.data) > 0


    def clear(self) -> None:
        self.data.clear()


    def __iter__(self):
        return iter(self.data)
