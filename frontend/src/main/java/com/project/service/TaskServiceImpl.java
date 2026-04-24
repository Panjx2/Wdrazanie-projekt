package com.project.service;

import com.project.exception.HttpException;
import com.project.model.Task;
import com.project.model.TaskStatus;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

@Service
public class TaskServiceImpl implements TaskService {
    private static final Logger logger = LoggerFactory.getLogger(TaskServiceImpl.class);
    private final RestClient restClient;

    public TaskServiceImpl(RestClient restClient) {
        this.restClient = restClient;
    }

    private String getResourcePath() {
        return "/api/tasks";
    }

    private String getResourcePath(Long id) {
        return String.format("%s/%d", getResourcePath(), id);
    }

    @Override
    public Task createTask(Long projectId, Task task) {
        String resourcePath = String.format("%s/project/%d", getResourcePath(), projectId);
        logger.info("REQUEST -> POST {}", resourcePath);
        return restClient.post()
                .uri(resourcePath)
                .accept(MediaType.APPLICATION_JSON)
                .body(task)
                .retrieve()
                .onStatus(HttpStatusCode::isError, (req, res) -> {
                    throw new HttpException(res.getStatusCode(), res.getHeaders());
                })
                .body(Task.class);
    }

    @Override
    public Task getTaskById(Long id) {
        String resourcePath = getResourcePath(id);
        logger.info("REQUEST -> GET {}", resourcePath);
        return restClient.get()
                .uri(resourcePath)
                .retrieve()
                .onStatus(HttpStatusCode::isError, (req, res) -> {
                    throw new HttpException(res.getStatusCode(), res.getHeaders());
                })
                .body(Task.class);
    }

    @Override
    public List<Task> getTasksByProject(Long projectId) {
        String resourcePath = String.format("%s/project/%d", getResourcePath(), projectId);
        logger.info("REQUEST -> GET {}", resourcePath);
        return restClient.get()
                .uri(resourcePath)
                .retrieve()
                .onStatus(HttpStatusCode::isError, (req, res) -> {
                    throw new HttpException(res.getStatusCode(), res.getHeaders());
                })
                .body(new ParameterizedTypeReference<List<Task>>() {});
    }

    @Override
    public Task updateTask(Long taskId, Task task) {
        String resourcePath = getResourcePath(taskId);
        logger.info("REQUEST -> PUT {}", resourcePath);
        return restClient.put()
                .uri(resourcePath)
                .accept(MediaType.APPLICATION_JSON)
                .body(task)
                .retrieve()
                .onStatus(HttpStatusCode::isError, (req, res) -> {
                    throw new HttpException(res.getStatusCode(), res.getHeaders());
                })
                .body(Task.class);
    }

    @Override
    public Task assignUser(Long taskId, Long userId) {
        String resourcePath = String.format("%s/%d/assign/%d", getResourcePath(), taskId, userId);
        logger.info("REQUEST -> PATCH {}", resourcePath);
        return restClient.patch()
                .uri(resourcePath)
                .retrieve()
                .onStatus(HttpStatusCode::isError, (req, res) -> {
                    throw new HttpException(res.getStatusCode(), res.getHeaders());
                })
                .body(Task.class);
    }

    @Override
    public Task changeStatus(Long taskId, TaskStatus status) {
        String resourcePath = String.format("%s/%d/status?status=%s", getResourcePath(), taskId, status.name());
        logger.info("REQUEST -> PATCH {}", resourcePath);
        return restClient.patch()
                .uri(resourcePath)
                .retrieve()
                .onStatus(HttpStatusCode::isError, (req, res) -> {
                    throw new HttpException(res.getStatusCode(), res.getHeaders());
                })
                .body(Task.class);
    }

    @Override
    public void deleteTask(Long taskId) {
        String resourcePath = getResourcePath(taskId);
        logger.info("REQUEST -> DELETE {}", resourcePath);
        restClient.delete()
                .uri(resourcePath)
                .retrieve()
                .onStatus(HttpStatusCode::isError, (req, res) -> {
                    throw new HttpException(res.getStatusCode(), res.getHeaders());
                })
                .toBodilessEntity();
    }
}
