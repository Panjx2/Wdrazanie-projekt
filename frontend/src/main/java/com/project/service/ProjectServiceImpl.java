package com.project.service;

import com.project.exception.HttpException;
import com.project.model.Project;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

@Service
public class ProjectServiceImpl implements ProjectService {
    private static final Logger logger = LoggerFactory.getLogger(ProjectServiceImpl.class);

    private final RestClient restClient;

    public ProjectServiceImpl(RestClient restClient) {
        this.restClient = restClient;
    }

    private String getResourcePath() {
        return "/api/projects";
    }

    private String getResourcePath(Long id) {
        return String.format("%s/%d", getResourcePath(), id);
    }

    @Override
    public List<Project> getAllProjects() {
        logger.info("REQUEST -> GET {}", getResourcePath());
        return restClient.get()
                .uri(getResourcePath())
                .retrieve()
                .body(new ParameterizedTypeReference<List<Project>>() {});
    }

    @Override
    public Project getProjectById(Long id) {
        String resourcePath = getResourcePath(id);
        logger.info("REQUEST -> GET {}", resourcePath);
        return restClient.get()
                .uri(resourcePath)
                .retrieve()
                .onStatus(HttpStatusCode::isError, (req, res) -> {
                    throw new HttpException(res.getStatusCode(), res.getHeaders());
                })
                .body(Project.class);
    }

    @Override
    public Project createProject(Project project) {
        logger.info("REQUEST -> POST {}", getResourcePath());
        return restClient.post()
                .uri(getResourcePath())
                .accept(MediaType.APPLICATION_JSON)
                .body(project)
                .retrieve()
                .onStatus(HttpStatusCode::isError, (req, res) -> {
                    throw new HttpException(res.getStatusCode(), res.getHeaders());
                })
                .body(Project.class);
    }

    @Override
    public Project updateProject(Long id, Project project) {
        String resourcePath = getResourcePath(id);
        logger.info("REQUEST -> PUT {}", resourcePath);
        return restClient.put()
                .uri(resourcePath)
                .accept(MediaType.APPLICATION_JSON)
                .body(project)
                .retrieve()
                .onStatus(HttpStatusCode::isError, (req, res) -> {
                    throw new HttpException(res.getStatusCode(), res.getHeaders());
                })
                .body(Project.class);
    }

    @Override
    public void deleteProject(Long id) {
        String resourcePath = getResourcePath(id);
        logger.info("REQUEST -> DELETE {}", resourcePath);
        restClient.delete()
                .uri(resourcePath)
                .retrieve()
                .onStatus(HttpStatusCode::isError, (req, res) -> {
                    throw new HttpException(res.getStatusCode(), res.getHeaders());
                })
                .toBodilessEntity();
    }

    @Override
    public Project addUserToProject(Long projectId, Long userId) {
        String resourcePath = String.format("%s/%d/users/%d", getResourcePath(), projectId, userId);
        logger.info("REQUEST -> POST {}", resourcePath);
        return restClient.post()
                .uri(resourcePath)
                .retrieve()
                .onStatus(HttpStatusCode::isError, (req, res) -> {
                    throw new HttpException(res.getStatusCode(), res.getHeaders());
                })
                .body(Project.class);
    }

    @Override
    public Project removeUserFromProject(Long projectId, Long userId) {
        String resourcePath = String.format("%s/%d/users/%d", getResourcePath(), projectId, userId);
        logger.info("REQUEST -> DELETE {}", resourcePath);
        return restClient.delete()
                .uri(resourcePath)
                .retrieve()
                .onStatus(HttpStatusCode::isError, (req, res) -> {
                    throw new HttpException(res.getStatusCode(), res.getHeaders());
                })
                .body(Project.class);
    }
}