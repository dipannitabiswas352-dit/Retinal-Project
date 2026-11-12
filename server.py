# ============================================================
# FLOWER FEDERATED LEARNING SERVER
# ============================================================

import os

import flwr as fl
import matplotlib.pyplot as plt
import torch

from flwr.common import parameters_to_ndarrays


# ============================================================
# GLOBAL STORAGE
# ============================================================

LAST_PARAMETERS = None

round_history = []
loss_history = []
accuracy_history = []


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULTS_DIR = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "results"
    )
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# CUSTOM FEDAVG STRATEGY
# ============================================================

class StableFedAvg(
    fl.server.strategy.FedAvg
):

    def aggregate_fit(
        self,
        server_round,
        results,
        failures
    ):

        global LAST_PARAMETERS

        print("\n")
        print("=" * 60)
        print(
            f"FEDERATED LEARNING - ROUND "
            f"{server_round}"
        )
        print("=" * 60)

        print(
            f"Clients received : "
            f"{len(results)}"
        )

        print(
            f"Failures         : "
            f"{len(failures)}"
        )

        # ----------------------------------------------------
        # CLIENT FAILURES
        # ----------------------------------------------------

        if failures:

            print("\nCLIENT FAILURES:")

            for failure in failures:

                print(
                    f"  {failure}"
                )

        # ----------------------------------------------------
        # NO RESULTS
        # ----------------------------------------------------

        if not results:

            print(
                "\nNO CLIENT RESULTS RECEIVED"
            )

            return None, {}

        # ----------------------------------------------------
        # FLOWER FEDAVG
        # ----------------------------------------------------

        aggregated = super().aggregate_fit(
            server_round,
            results,
            failures
        )

        if aggregated is None:

            print(
                "\nWARNING:"
            )

            print(
                "Flower aggregation returned None."
            )

            return None, {}

        (
            parameters_aggregated,
            metrics_aggregated
        ) = aggregated

        # ----------------------------------------------------
        # SAVE CURRENT GLOBAL PARAMETERS
        # ----------------------------------------------------

        LAST_PARAMETERS = (
            parameters_aggregated
        )

        print(
            "\nGlobal parameters "
            "aggregated successfully."
        )

        print(
            f"Federated tensors: "
            f"{len(parameters_aggregated.tensors)}"
        )

        # ----------------------------------------------------
        # CLIENT METRICS
        # ----------------------------------------------------

        total_examples = 0

        total_loss = 0.0
        total_accuracy = 0.0

        loss_examples = 0
        accuracy_examples = 0

        # ----------------------------------------------------
        # PROCESS CLIENT RESULTS
        # ----------------------------------------------------

        for client, fit_res in results:

            num_examples = (
                fit_res.num_examples
            )

            print("\n" + "-" * 50)

            print(
                f"Client {client.cid}"
            )

            print(
                f"Training samples : "
                f"{num_examples}"
            )

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            if fit_res.metrics:

                # --------------------------------------------
                # LOSS
                # --------------------------------------------

                if "train_loss" in fit_res.metrics:

                    try:

                        client_loss = float(
                            fit_res.metrics[
                                "train_loss"
                            ]
                        )

                        print(
                            f"Client loss      : "
                            f"{client_loss:.4f}"
                        )

                        total_loss += (
                            client_loss
                            * num_examples
                        )

                        loss_examples += (
                            num_examples
                        )

                    except Exception as e:

                        print(
                            "Could not read "
                            "client loss:"
                        )

                        print(e)

                # --------------------------------------------
                # ACCURACY
                # --------------------------------------------

                if "train_accuracy" in fit_res.metrics:

                    try:

                        client_accuracy = float(
                            fit_res.metrics[
                                "train_accuracy"
                            ]
                        )

                        print(
                            f"Client accuracy  : "
                            f"{client_accuracy:.4f}"
                        )

                        total_accuracy += (
                            client_accuracy
                            * num_examples
                        )

                        accuracy_examples += (
                            num_examples
                        )

                    except Exception as e:

                        print(
                            "Could not read "
                            "client accuracy:"
                        )

                        print(e)

            total_examples += num_examples

        # ----------------------------------------------------
        # GLOBAL TRAINING METRICS
        # ----------------------------------------------------

        global_loss = None
        global_accuracy = None

        if loss_examples > 0:

            global_loss = (
                total_loss /
                loss_examples
            )

            loss_history.append(
                global_loss
            )

        if accuracy_examples > 0:

            global_accuracy = (
                total_accuracy /
                accuracy_examples
            )

            accuracy_history.append(
                global_accuracy
            )

        # ----------------------------------------------------
        # ROUND HISTORY
        # ----------------------------------------------------

        round_history.append(
            server_round
        )

        # ----------------------------------------------------
        # PRINT GLOBAL METRICS
        # ----------------------------------------------------

        print("\n" + "-" * 50)

        if global_loss is not None:

            print(
                f"Global Train Loss : "
                f"{global_loss:.4f}"
            )

        else:

            print(
                "Global Train Loss : N/A"
            )

        if global_accuracy is not None:

            print(
                f"Global Train Acc  : "
                f"{global_accuracy:.4f}"
            )

        else:

            print(
                "Global Train Acc  : N/A"
            )

        print(
            f"Total samples     : "
            f"{total_examples}"
        )

        print("\n" + "=" * 60)

        print(
            f"ROUND {server_round} "
            f"COMPLETED SUCCESSFULLY"
        )

        print("=" * 60)

        return (
            parameters_aggregated,
            metrics_aggregated
        )


# ============================================================
# SAVE GLOBAL MODEL
# ============================================================

def save_global_model():

    global LAST_PARAMETERS

    print("\n")
    print("=" * 60)
    print("SAVING GLOBAL FEDERATED MODEL")
    print("=" * 60)

    if LAST_PARAMETERS is None:

        print("\nWARNING:")

        print(
            "No global parameters were captured."
        )

        print(
            "global_model.pth was NOT created."
        )

        print("=" * 60)

        return False

    # --------------------------------------------------------
    # CONVERT FLOWER PARAMETERS
    # --------------------------------------------------------

    try:

        tensors = parameters_to_ndarrays(
            LAST_PARAMETERS
        )

    except Exception as e:

        print(
            "\nERROR converting "
            "Flower parameters:"
        )

        print(e)

        return False

    if len(tensors) == 0:

        print("\nERROR:")

        print(
            "No federated tensors were received."
        )

        return False

    # --------------------------------------------------------
    # PRINT SHAPES
    # --------------------------------------------------------

    print(
        f"\nFederated tensors received: "
        f"{len(tensors)}"
    )

    print("\nTensor information:")

    for i, tensor in enumerate(tensors):

        print(
            f"  Tensor {i + 1}: "
            f"shape={tensor.shape}, "
            f"dtype={tensor.dtype}"
        )

    # --------------------------------------------------------
    # OUTPUT PATH
    # --------------------------------------------------------

    output_path = os.path.join(
        RESULTS_DIR,
        "global_model.pth"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    try:

        torch.save(
            tensors,
            output_path
        )

    except Exception as e:

        print(
            "\nERROR saving global model:"
        )

        print(e)

        return False

    # --------------------------------------------------------
    # VERIFY
    # --------------------------------------------------------

    if not os.path.exists(
        output_path
    ):

        print(
            "\nERROR:"
        )

        print(
            "global_model.pth was not created."
        )

        return False

    file_size = os.path.getsize(
        output_path
    )

    print("\n")
    print("=" * 60)
    print("GLOBAL MODEL SAVED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"Path:\n{output_path}"
    )

    print(
        f"\nFederated tensors saved: "
        f"{len(tensors)}"
    )

    print(
        f"File size: "
        f"{file_size / 1024:.2f} KB"
    )

    print("\nTensor shapes:")

    for i, tensor in enumerate(tensors):

        print(
            f"  Tensor {i + 1}: "
            f"{tensor.shape}"
        )

    print("=" * 60)

    return True


# ============================================================
# SAVE TRAINING GRAPHS
# ============================================================

def save_training_graphs():

    print("\n")
    print("=" * 60)
    print("SAVING TRAINING GRAPHS")
    print("=" * 60)

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    if loss_history:

        loss_path = os.path.join(
            RESULTS_DIR,
            "fl_loss_curve.png"
        )

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            round_history[:len(loss_history)],
            loss_history,
            marker="o"
        )

        plt.xlabel(
            "Federated Learning Round"
        )

        plt.ylabel(
            "Training Loss"
        )

        plt.title(
            "Federated Learning Training Loss"
        )

        plt.grid(True)

        plt.savefig(
            loss_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "\nLoss graph saved:"
        )

        print(
            loss_path
        )

    else:

        print(
            "\nNo loss history available."
        )

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    if accuracy_history:

        accuracy_path = os.path.join(
            RESULTS_DIR,
            "fl_accuracy_curve.png"
        )

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            round_history[:len(accuracy_history)],
            accuracy_history,
            marker="o"
        )

        plt.xlabel(
            "Federated Learning Round"
        )

        plt.ylabel(
            "Training Accuracy"
        )

        plt.title(
            "Federated Learning Training Accuracy"
        )

        plt.grid(True)

        plt.savefig(
            accuracy_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "\nAccuracy graph saved:"
        )

        print(
            accuracy_path
        )

    else:

        print(
            "\nNo accuracy history available."
        )

    print("\n" + "=" * 60)


# ============================================================
# VERIFY GLOBAL MODEL
# ============================================================

def verify_global_model():

    output_path = os.path.join(
        RESULTS_DIR,
        "global_model.pth"
    )

    print("\n")
    print("=" * 60)
    print("VERIFYING GLOBAL MODEL FILE")
    print("=" * 60)

    if not os.path.exists(
        output_path
    ):

        print("\nERROR:")

        print(
            "global_model.pth does not exist."
        )

        return False

    try:

        tensors = torch.load(
            output_path,
            map_location="cpu"
        )

    except Exception as e:

        print(
            "\nERROR verifying file:"
        )

        print(e)

        return False

    if not isinstance(
        tensors,
        (list, tuple)
    ):

        print("\nERROR:")

        print(
            "Saved model file does not "
            "contain a tensor list."
        )

        return False

    if len(tensors) == 0:

        print("\nERROR:")

        print(
            "Saved tensor list is empty."
        )

        return False

    print(
        f"\nLoaded tensors: "
        f"{len(tensors)}"
    )

    print("\nTensor shapes:")

    for i, tensor in enumerate(tensors):

        print(
            f"  Tensor {i + 1}: "
            f"shape={tensor.shape}"
        )

    print(
        "\nGLOBAL MODEL FILE "
        "VERIFICATION PASSED"
    )

    print("=" * 60)

    return True


# ============================================================
# MAIN SERVER
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print(
        "STARTING FLOWER FEDERATED "
        "LEARNING SERVER"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # SERVER SETTINGS
    # --------------------------------------------------------

    SERVER_ADDRESS = "127.0.0.1:9090"

    NUM_ROUNDS = 10

    # --------------------------------------------------------
    # IMPORTANT:
    # Currently testing ONE client at a time.
    # --------------------------------------------------------

    MIN_CLIENTS = 1

    print(
        f"Server address   : "
        f"{SERVER_ADDRESS}"
    )

    print(
        f"Federated rounds : "
        f"{NUM_ROUNDS}"
    )

    print(
        f"Minimum clients  : "
        f"{MIN_CLIENTS}"
    )

    print(
        "\nCurrent test mode:"
    )

    print(
        "ONE CLIENT AT A TIME"
    )

    print(
        "\nResults directory:\n"
        f"{RESULTS_DIR}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # FEDAVG
    # --------------------------------------------------------

    strategy = StableFedAvg(

        fraction_fit=1.0,

        fraction_evaluate=1.0,

        min_fit_clients=MIN_CLIENTS,

        min_available_clients=MIN_CLIENTS,

        min_evaluate_clients=MIN_CLIENTS,

        accept_failures=False,
    )

    # --------------------------------------------------------
    # START SERVER
    # --------------------------------------------------------

    try:

        fl.server.start_server(

            server_address=SERVER_ADDRESS,

            config=fl.server.ServerConfig(
                num_rounds=NUM_ROUNDS
            ),

            strategy=strategy
        )

    except KeyboardInterrupt:

        print(
            "\n\nServer stopped by user."
        )

    except Exception as e:

        print("\n")
        print("=" * 60)
        print("SERVER ERROR")
        print("=" * 60)

        print(e)

        print("=" * 60)

        raise

    # --------------------------------------------------------
    # TRAINING FINISHED
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FEDERATED TRAINING FINISHED")
    print("=" * 60)

    print(
        f"Completed rounds: "
        f"{len(round_history)}"
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_saved = save_global_model()

    # --------------------------------------------------------
    # SAVE GRAPHS
    # --------------------------------------------------------

    save_training_graphs()

    # --------------------------------------------------------
    # VERIFY MODEL
    # --------------------------------------------------------

    if model_saved:

        verify_global_model()

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("SERVER EXECUTION COMPLETED")
    print("=" * 60)

    if model_saved:

        print(
            "\nSUCCESS:"
        )

        print(
            "global_model.pth is ready."
        )

        print(
            "\nCheck with:"
        )

        print(
            "Test-Path "
            "..\\results\\global_model.pth"
        )

    else:

        print(
            "\nWARNING:"
        )

        print(
            "Global model was NOT saved."
        )

    print("=" * 60)