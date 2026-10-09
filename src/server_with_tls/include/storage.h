#include <definitions.h>

size_t get_array_size(void **array);

table *init_table(int capacity);

char *insert_into_table(table *t, model *m);

void resize_table(table *t, int index, model *m);

bool contains_key(table *t, char *id);

char *find_duplicate_names_from_id(table *t, char **names);

model *get_model(table *t, char *id);

int remove_model_from_table(table *t, char *id);

void free_inference_model_ptr(void *inference_model_ptr, ModelType type);

void free_inference_models_ptr(void **inference_models_ptr, int length, ModelType type);

void deallocate_model(model *current);

void free_table(table* t);

void print_table(table *t);

operator_io **init_operator_io(int length);

void resize_operators_io(operator_io ***io, int length, int index);

void insert_into_operator_io(operator_io ***io, operator_io *input, int index, char *name);

void free_operator_io(operator_io **io);

void print_operator_io(operator_io **io);

operator_node *create_operator_node(char *model_name, int node_id);

void insert_parent_to_operator_node(operator_node *parent, operator_node *child);

void insert_child_to_operator_node(operator_node *parent, operator_node *child);

void update_node(operator_io **io, int id, operator_node *head);

operator_node *search_operator_node_by_name(operator_node *node, const char *target_name);

void reset_node_visibility(operator_node *node);

void free_operator_node_info(operator_node *node);

void free_operator_node(operator_node *node);

void print_operator_node(operator_node *node);
