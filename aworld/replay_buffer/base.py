import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, TypeVar, Union
from abc import ABC, abstractmethod

from aworld.core.common import ActionModel, Observation

T = TypeVar('T')
QueryCondition = Union[Dict[str, Any], List[Dict[str, any]]]

@dataclass
class Experience:
    '''
    Experience of agent.
    '''
    state: Observation
    action: ActionModel
    reward_t: float
    adv_t: float
    v_t: float


@dataclass
class ExpMeta:
    '''
    Experience meta data.
    '''
    task_id: str
    task_name: str
    agent_id: str
    step: int
    timestamp: float


@dataclass
class DataRow:
    '''
    Data row for storing data.
    '''
    exp_meta: ExpMeta
    exp_data: Experience
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


class Storage(ABC):
    '''
    Storage for storing and sampling data.
    '''

    @abstractmethod
    def add(self, data: DataRow):
        '''
        Add data to the storage.
        Args:
            data (DataRow): Data to add.
        '''

    @abstractmethod
    def add_batch(self, data_batch: List[DataRow]):
        '''
        Add batch of data to the storage.
        Args:
            data_batch (List[DataRow]): List of data to add.
        '''

    @abstractmethod
    def size(self, query_contion: QueryCondition) -> int:
        '''
        Get the size of the storage.
        Returns:
            int: Size of the storage.
        '''

    @abstractmethod
    def get_paginated(self, page: int, page_size: int, query_contion: QueryCondition) -> List[DataRow]:
        '''
        Get paginated data from the storage.
        Args:
            page (int): Page number.
            page_size (int): Number of data per page.
        Returns:
            List[DataRow]: List of data.
        '''

    @abstractmethod
    def get_all(self) -> List[DataRow]:
        '''
        Get all data from the storage.
        Returns:
            List[DataRow]: List of data.
        '''

    @abstractmethod
    def get_by_task_ids(self, task_id: str) -> List[DataRow]:
        '''
        Get data by task_id from the storage.
        Args:
            task_id (str): Task id.
        Returns:
            List[DataRow]: List of data.
        '''

    @abstractmethod
    def get_bacth_by_task_ids(self, task_ids: List[str]) -> Dict[str, List[DataRow]]:
        '''
        Get batch of data by task_ids from the storage.
        Args:
            task_ids (List[str]): List of task ids.
        Returns:
            Dict[str, List[DataRow]]: Dictionary of data.
            The key is the task_id and the value is the list of data.
            The list of data is sorted by step.
        '''


class Sampler(ABC):
    '''
    Sample data from the storage.
    '''

    def sorted_by_step(self, task_experience: List[DataRow]) -> List[DataRow]:
        '''
        Sort the task experience by step.
        Args:
            task_experience (List[DataRow]): List of task experience.
        Returns:
            List[DataRow]: List of task experience sorted by step.
        '''
        return sorted(task_experience, key=lambda x: x.step)

    def sample(self, storage: Storage, batch_size: int) -> Dict[str, List[DataRow]]:
        '''
        Sample data from the storage.
        Args:
            storage (Storage): Storage to sample from.
            batch_size (int): Number of data to sample.
        Returns:
            Dict[str, List[DataRow]]: Dictionary of sampled data.
            The key is the task_id and the value is the list of data.
            The list of data is sorted by step.
        '''
        task_ids = self.sample_task_ids(storage, batch_size)
        raws = storage.get_bacth_by_task_ids(task_ids)
        return {task_id: self.sorted_by_step(raws) for task_id, raws in raws.items()}

    @abstractmethod
    def sample_task_ids(self, storage: Storage, batch_size: int) -> List[str]:
        '''
        Sample task_ids from the storage.
        Args:
            storage (Storage): Storage to sample from.
            batch_size (int): Number of task_ids to sample.
        Returns:
            List[str]: List of task_ids.
        '''


class Converter(ABC):
    '''
    Convert data to dataset row.
    '''

    @abstractmethod
    def to_dataset_row(self, task_experience: List[DataRow]) -> T:
        '''
        Convert task experience to dataset row.
        Args:
            task_experience (List[DataRow]): List of task experience.
        Returns:
            T: type of dataset row.
        '''


class InMemoryStorage(Storage):
    '''
    In-memory storage for storing and sampling data.
    '''

    def __init__(self, max_capacity: int = 10000):
        self._data: Dict[str, List[DataRow]] = {}
        self._max_capacity = max_capacity
        self._fifo_queue = []  # (task_id)

    def add(self, data: DataRow):
        if not data:
            raise ValueError("Data is required")
        if not data.exp_meta:
            raise ValueError("exp_meta is required")

        while self.size() >= self._max_capacity and self._fifo_queue:
            oldest_task_id = self._fifo_queue.pop(0)
            if oldest_task_id in self._data:
                del self._data[oldest_task_id]

        if data.exp_meta.task_id not in self._data:
            self._data[data.exp_meta.task_id] = []
        self._data[data.exp_meta.task_id].append(data)
        self._fifo_queue.append(data.exp_meta.task_id)

        if data.exp_meta.task_id not in self._data:
            self._data[data.exp_meta.task_id] = []
        self._data[data.exp_meta.task_id].append(data)

    def add_batch(self, data_batch: List[DataRow]):
        for data in data_batch:
            self.add(data)

    def size(self) -> int:
        return sum([len(data) for data in self._data.values()])

    def get_paginated(self, page: int, page_size: int) -> List[DataRow]:
        if page < 1:
            raise ValueError("Page must be greater than 0")
        if page_size < 1:
            raise ValueError("Page size must be greater than 0")
        all_data = self.get_all()
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        return all_data[start_index:end_index]

    def get_all(self) -> List[DataRow]:
        all_data = []
        for data in self._data.values():
            all_data.extend(data)
        return all_data

    def get_by_task_ids(self, task_id: str) -> List[DataRow]:
        return self._data.get(task_id, [])

    def get_bacth_by_task_ids(self, task_ids: List[str]) -> Dict[str, List[DataRow]]:
        return {task_id: self._data.get(task_id, []) for task_id in task_ids}

    def clear(self):
        self._data = {}
        self._fifo_queue = []


class RandomSample(Sampler):
    '''
    Randomly sample data from the storage.
    '''

    def sample_task_ids(self, storage: Storage, batch_size: int) -> List[str]:
        total_size = storage.size()
        if total_size <= batch_size:
            return storage.get_all()

        sampled_task_ids = set()
        page_size = min(100, batch_size * 2)
        total_pages = (total_size + page_size - 1)
        visited_pages = set()
        while len(sampled_task_ids) < batch_size and len(visited_pages) < total_pages:
            page = random.choice(
                [p for p in range(1, total_pages+1) if p not in visited_pages])
            visited_pages.add(page)

            current_page = storage.get_paginated(page, page_size)
            if not current_page:
                continue
            current_page_task_ids = set(
                [data.exp_meta.task_id for data in current_page if data.exp_meta.task_id not in sampled_task_ids])
            sample_count = min(len(current_page_task_ids),
                               batch_size - len(sampled_task_ids))
            sampled_task_ids.update(random.sample(
                current_page_task_ids, sample_count))

        return list(sampled_task_ids)


class DefaultConverter(Converter):
    '''
    Default converter do nothing.
    '''

    def to_dataset_row(self, task_experience: List[DataRow]) -> List[DataRow]:
        return task_experience


class ReplayBuffer:
    '''
    Replay buffer for storing and sampling data.
    '''

    def __init__(
        self,
        storage: Storage = InMemoryStorage()
    ):
        self._storage = storage

    def store(self, data: DataRow):
        '''
        Store data in the replay buffer.
        '''
        if not data:
            raise ValueError("Data is required")
        self._storage.add(data)

    def store_batch(self, data_batch: List[DataRow]):
        '''
        Store batch of data in the replay buffer.
        '''
        if not data_batch:
            raise ValueError("Data batch is required")
        self._storage.add_batch(data_batch)

    def sample_and_convert(self,
                           sampler: Sampler = RandomSample(),
                           converter: Converter = DefaultConverter(),
                           batch_size: int = 1000) -> List[T]:
        '''
        Sample data from the replay buffer and convert to dataset row.
        DefaultConverter return List[DataRow]
        '''

        sampled_data = sampler.sample(self._storage, batch_size)
        return [converter.to_dataset_row(task_experiences) for task_experiences in sampled_data.values()]

class QueryBuilder:
    '''
    Query builder for replay buffer. result example:
    {
        "and": [
            {"field": "field1", "value": "value1", "op": "eq"}, 
            {"or": [{"field": "field2", "value": "value2", "op": "eq"}, {"field": "field3", "value": "value3", "op": "eq"}]}
        ]
    }
    '''

    def __init__(self) -> None:
        self.conditions: List[Dict[str, any]] = []
        self.logical_ops: List[str] = []

    def eq(self, field: str, value: any) -> 'QueryBuilder':
        self.conditions.append({"field": field, "value": value, "op": "eq"})
        return self

    def ne(self, field: str, value: any) -> 'QueryBuilder':
        self.conditions.append({"field": field, "value": value, "op": "ne"})
        return self

    def gt(self, field: str, value: any) -> 'QueryBuilder':
        self.conditions.append({"field": field, "value": value, "op": "gt"})
        return self

    def gte(self, field: str, value: any) -> 'QueryBuilder':
        self.conditions.append({"field": field, "value": value, "op": "gte"})
        return self

    def lt(self, field: str, value: any) -> 'QueryBuilder':
        self.conditions.append({"field": field, "value": value, "op": "lt"})
        return self

    def lte(self, field: str, value: any) -> 'QueryBuilder':
        self.conditions.append({"field": field, "value": value, "op": "lte"})
        return self

    def in_(self, field: str, value: any) -> 'QueryBuilder':
        self.conditions.append({"field": field, "value": value, "op": "in"})
        return self

    def not_in(self, field: str, value: any) -> 'QueryBuilder':
        self.conditions.append(
            {"field": field, "value": value, "op": "not_in"})
        return self

    def like(self, field: str, value: any) -> 'QueryBuilder':
        self.conditions.append({"field": field, "value": value, "op": "like"})
        return self

    def not_like(self, field: str, value: any) -> 'QueryBuilder':
        self.conditions.append(
            {"field": field, "value": value, "op": "not_like"})
        return self

    def is_null(self, field: str) -> 'QueryBuilder':
        self.conditions.append({"field": field, "op": "is_null"})
        return self

    def is_not_null(self, field: str) -> 'QueryBuilder':
        self.conditions.append({"field": field, "op": "is_not_null"})
        return self

    def and_(self) -> 'QueryBuilder':
        self.logical_ops.append("and")
        return self

    def or_(self) -> 'QueryBuilder':
        self.logical_ops.append("or")
        return self

    def nested(self, builder: 'QueryBuilder') -> 'QueryBuilder':
        self.conditions.append({"nested": builder.build()})
        return self

    def build(self) -> QueryCondition:
        conditions = self.conditions  # all conditions（including nested）
        operators = self.logical_ops

        # Validate condition and operator counts (n conditions need n-1 operators)
        if len(operators) != len(conditions) - 1:
            raise ValueError("Mismatch between condition and operator counts")

        # Use stack to handle operator precedence (simplified version supporting and/or)
        stack: List[Union[Dict[str, any], str]] = []

        for i, item in enumerate(conditions):
            if i == 0:
                # First element goes directly to stack (condition or nested)
                stack.append(item)
                continue

            # Pop stack top as left operand
            left = stack.pop()
            op = operators[i-1]       # Current operator (and/or)
            right = item              # Right operand (current condition)

            # Build logical expression: {op: [left, right]}
            expr = {op: [left, right]}
            # Push result back to stack for further operations
            stack.append(expr)

        # Process nested conditions (recursive unfolding)
        def process_nested(cond: any) -> any:
            if isinstance(cond, dict):
                if "nested" in cond:
                    # Recursively process sub-conditions
                    return process_nested(cond["nested"])
                # Recursively process child elements
                return {k: process_nested(v) for k, v in cond.items()}
            elif isinstance(cond, list):
                return [process_nested(item) for item in cond]
            return cond

        # Final result: only one element left in stack, return after processing nested
        result = stack[0] if stack else None
        return process_nested(result) if result else None

