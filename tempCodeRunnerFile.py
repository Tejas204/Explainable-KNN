for k_value in range(1, range_of_K+1):
#     print(f"Computing for k: {k_value}")
#     predicted_labels = []
#     error_rates = []
#     model = KNN(k=k_value, batch_size=batch_size)
#     for i in range(10):
#         prediction = model.compute_knn(query=test_dataset[i][0])
#         predicted_labels.append(prediction)

#     error = model.compute_error_rate(predicted_labels, test_dataset[:][1])
#     print(f"Error for k = {k_value} is: {error}\n")
#     error_rates.append(error)

# # Visualize
# plt.plot([value+1 for value in range(range_of_K)], error_rates)
# plt.xlabel("K-values")
# plt.ylabel("Error rates")
# plt.title("K-values vs Error rate")
# plt.show()