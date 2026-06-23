import torch
import data
import model as model_definition
import storage
import os

"""
TO DO:
Existing bug where the top predicted entity is always the head entity.
This may be due to the fact that the training file is very small, and therefore the model is not learning meaningful embeddings.
    - Try to run this with our larger movielens dataset and see if the issue persists.
"""

# Predict tail entities for a given head entity and relation for k positions

def predict_sequence(model, h, r, entity2id, relation2id, top_k=None):
    if top_k is None:
        # Ask user for k value
        top_k = int(input("Enter number of top predictions to display: ").strip())
    id2entity = {v: k for k, v in entity2id.items()}

    h_id = entity2id[h]
    r_id = relation2id[r]

    # Create entity IDs tensors
    num_entities = len(entity2id)
    entity_ids = torch.arange(num_entities)

    # Create head and relation tensors
    heads = torch.full((num_entities,), h_id)
    relations = torch.full((num_entities,), r_id)

    # Create triplets tensor
    triplets = torch.stack((heads, relations, entity_ids), dim=1)

    # Predict scores for potential tail entities
    scores = model.predict(triplets)

    # Sort scores
    sorted_indices = torch.argsort(scores)

    print(f"\nQuery head: ({h})")
    print(f"Query relation: ({r})\n")
    print("Top predictions (tail entities):")

    # Display k highest scoring prediction
    for rank, idx in enumerate(sorted_indices[:top_k], start=1):
        print(f"{rank}. {id2entity[idx.item()]}")


# Load the model and mappings from the dataset/checkpoint
def load_model(dataset_path, checkpoint_path):
    train_path = os.path.join(dataset_path, "train.txt")

    # Create entity and relation mappings
    entity2id, relation2id = data.create_mappings(train_path)

    # Init TransE model
    model = model_definition.TransE(
        entity_count=len(entity2id),
        relation_count=len(relation2id),
        dim=50,
        margin=1.0,
        device=torch.device("cpu"),
        norm=1
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Load model checkpoint
    storage.load_checkpoint(checkpoint_path, model, optimizer)

    model.eval()

    # Return the model and mappings
    return model, entity2id, relation2id


def main():
    # Set paths for dataset and checkpoint
    dataset_path = "./test_data"
    checkpoint_path = "checkpoint.tar"

    # Load the model and mappings
    model, entity2id, relation2id = load_model(dataset_path, checkpoint_path)

    print("TransE Prediction CLI\n")
    print("'exit' to quit.")

    while True:
        h = input("Enter head entity: ").strip()
        if h.lower() == "exit":
            break

        r = input("Enter relation: ").strip()
        if r.lower() == "exit":
            break

        # Call prediction function
        predict_sequence(model, h, r, entity2id, relation2id)


if __name__ == "__main__":
    main()