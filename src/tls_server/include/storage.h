#include <definitions.h>

size_t get_array_size(void **array);

onnx_table *init_onnx_table(int capacity);

char *insert_into_table(onnx_table *table, model *m);

void resize_table(onnx_table *table, int index, model *m);

bool contains_key(onnx_table *table, char *id);

char *find_duplicate_names_from_id(onnx_table *table, char **names);

model *get_model(onnx_table *table, char *id);

int remove_model_from_table(onnx_table *table, char *id);

void free_inference_model(TractInferenceModel *inference_model);

void free_inference_models(TractInferenceModel **inference_models, int length);

void deallocate_model(model *current);

void free_onnx_table(onnx_table* table);

void print_table(onnx_table *table);

operator_io **init_operator_io(int length);

void resize_operators_io(operator_io ***io, int length, int index);

void insert_into_operator_io(operator_io ***io, operator_io *input, int index, char *name);

int *find_index_of_operator_io(operator_io **io, int id);

void free_operator_io(operator_io **io);

void print_operator_io(operator_io **io);

operator_node *create_operator_node(char *model_name);

void insert_parent_to_operator_node(operator_node *parent, operator_node *child);

void insert_child_to_operator_node(operator_node *parent, operator_node *child);

void update_node(operator_io **io, int id, operator_node *head);

bool is_node_visited(operator_node *node, char **visited_nodes, int visited_count);

operator_node *search_operator_node_by_name(operator_node *node, const char *target_name);

void free_operator_node(operator_node *node, char **visited_nodes, int *visited_count);

void print_operator_node(operator_node *node, char **visited_nodes, int *visited_count);
