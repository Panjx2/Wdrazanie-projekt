package com.project.service;

import com.project.model.Task;
import com.project.model.TaskStatus;
import java.util.List;

public interface TaskService {
    Task createTask(Long projectId, Task task);
    Task getTaskById(Long id);
    List<Task> getTasksByProject(Long projectId);
    Task updateTask(Long taskId, Task task);
    Task assignUser(Long taskId, Long userId);
    Task changeStatus(Long taskId, TaskStatus status);
    void deleteTask(Long taskId);
}
