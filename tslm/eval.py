import typer
import datasets
from tslm.reward import compute_score_cascade

def main(
    data_path: str,
    workers: int = 10,
    num_tries: int = 1,
    score_threshold: float = 0.9,
    return_dataset: bool = False,
    clear_cache: bool = True,
):
    dataset = datasets.load_dataset("parquet", data_files={'test': data_path})['test']
    if clear_cache:
        dataset.cleanup_cache_files()

    def map_compute_score_fn(data: dict) -> dict:
        """Compute the reward score for the first num_tries responses."""
        responses_reward_scores = [
            compute_score_cascade(
                solution_str=response, 
                ground_truth=data['reward_model']['ground_truth'],
                do_print=False,
            )
            for response in data['responses'][:num_tries]
        ]
        responses_reward_scores = list(map(float, responses_reward_scores))
        any_correct = any([score > score_threshold for score in responses_reward_scores])

        # Update the data with the reward scores and correct responses
        data.update({
            'responses_reward_scores': responses_reward_scores,
            'any_correct': any_correct,
        })
        return data

    dataset = dataset.map(map_compute_score_fn, num_proc=workers)
    accuracy = len(dataset.filter(lambda x: x['any_correct'])) / len(dataset)
    print(f'Accuracy@{num_tries}: {accuracy:.3f}')
    if return_dataset:
        return dataset, accuracy
    return accuracy

if __name__ == "__main__":
    typer.run(main)
