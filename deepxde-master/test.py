def plot_best_state(train_state):
    X_train, y_train, X_test, y_test, best_y, best_ystd = train_state.packed_data()

    y_dim = y_train.shape[1]

    # Regression plot
    plt.figure()
    idx = np.argsort(X_test[:, 0])
    X = X_test[idx, 0]
    for i in range(y_dim):
        plt.plot(X_train[:, 0], y_train[:, i], "ok", label="Train")
        plt.plot(X, y_test[idx, i], ".k", label="True")
        plt.plot(X, best_y[idx, i], "or", label="Prediction")
        if best_ystd is not None:
            plt.plot(X, best_y[idx, i] + 2 * best_ystd[idx, i], "-b", label="95% CI")
            plt.plot(X, best_y[idx, i] - 2 * best_ystd[idx, i], "-b")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    # Residual plot
    plt.figure()
    residual = y_test[:, 0] - best_y[:, 0]
    plt.plot(best_y[:, 0], residual, "o", zorder=1)
    plt.hlines(0, plt.xlim()[0], plt.xlim()[1], linestyles="dashed", zorder=2)
    plt.xlabel("Predicted")
    plt.ylabel("Residual = Observed - Predicted")
    plt.tight_layout()

    if best_ystd is not None:
        plt.figure()
        for i in range(y_dim):
            plt.plot(X_test[:, 0], best_ystd[:, i], "-b")
            plt.plot(
                X_train[:, 0],
                np.interp(X_train[:, 0], X_test[:, 0], best_ystd[:, i]),
                "ok",
            )
        plt.xlabel("x")
        plt.ylabel("std(y)")
        
plot_best_state(train_state)