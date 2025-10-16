from typing import List, Any, Callable


class DataBatch:

    def __init__(self, batch_size: int, data_store_type: Callable=list):
        self._data_store_type = data_store_type
        self._data = data_store_type([])
        # used only when returning items of the set object. 
        self._cached_data = []
        self._batch_size = batch_size


    @property
    def data_store_type(self):
        return self._data_store_type


    def add(self, item: Any | List[Any]) -> None:
        """
        Add an item or list of items to the data collection.
        
        Args:
            item: a single item or list of items to add.
        """
        if isinstance(self._data, set):
            self._add_to_set(item)
        else:
            self._add_to_sequence(item)


    def _add_to_set(self, item: Any | List[Any]) -> None:
        
        if isinstance(item, list):
            self._data.update(item)
            self._cached_data.extend(item)
        else:
            self._data.add(item)
            self._cached_data.append(item)


    def _add_to_sequence(self, item: Any | List[Any]) -> None:
        
        if isinstance(item, list):
            self._data.extend(item)
        else:
            self._data.append(item)
        

    def is_full(self) -> bool:
        return len(self._data) >= self._batch_size
    

    def has_items(self) -> bool:
        return len(self._data) > 0


    def clear(self) -> None:
        self._data.clear()
        if self._cached_data:
            self._cached_data.clear()


    def __len__(self) -> int:
        return len(self._data)


    def __iter__(self):
        return iter(self._data)
    

    def __getitem__(self, idx) -> Any:
        if isinstance(self._data, set):
            return sorted(self._cached_data)[idx]
        return self._data[idx]
