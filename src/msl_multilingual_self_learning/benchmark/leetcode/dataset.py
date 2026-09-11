from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


DATASET_NAME = "newfacade/LeetCodeDataset"


@dataclass
class LeetCodeProblem:
     task_id: str
     question_id: int
     difficulty: str
     tags: list[str]

     problem_description: str
     starter_code: str
     entry_point: str
     test: str
     completion: str

     input_output: list[dict[str, str]]


def load_test_split() -> list[LeetCodeProblem]:
     dataset = load_dataset(
          DATASET_NAME,
          split="test",
     )

     problems: list[LeetCodeProblem] = []

     for row in dataset:
          problem = LeetCodeProblem(
               task_id=row["task_id"],
               question_id=row["question_id"],
               difficulty=row["difficulty"],
               tags=row["tags"],
               problem_description=row["problem_description"],
               starter_code=row["starter_code"],
               entry_point=row["entry_point"],
               test=row["test"],
               input_output=row["input_output"],
               completion=row["completion"],
          )

          problems.append(problem)

     return problems