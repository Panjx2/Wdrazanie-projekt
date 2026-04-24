package com.project.service;

import com.project.exception.HttpException;
import com.project.model.User;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

@Service
public class UserServiceImpl implements UserService {
    private static final Logger logger = LoggerFactory.getLogger(UserServiceImpl.class);

    private final RestClient restClient;

    public UserServiceImpl(RestClient restClient) {
        this.restClient = restClient;
    }

    private String getResourcePath() {
        return "/api/users";
    }

    private String getResourcePath(Long id) {
        return String.format("%s/%d", getResourcePath(), id);
    }

    @Override
    public User createUser(User user) {
        logger.info("REQUEST -> POST {}", getResourcePath());
        return restClient.post()
                .uri(getResourcePath())
                .accept(MediaType.APPLICATION_JSON)
                .body(user)
                .retrieve()
                .onStatus(HttpStatusCode::isError, (req, res) -> {
                    throw new HttpException(res.getStatusCode(), res.getHeaders());
                })
                .body(User.class);
    }

    @Override
    public List<User> getAllUsers() {
        logger.info("REQUEST -> GET {}", getResourcePath());
        return restClient.get()
                .uri(getResourcePath())
                .retrieve()
                .onStatus(HttpStatusCode::isError, (req, res) -> {
                    throw new HttpException(res.getStatusCode(), res.getHeaders());
                })
                .body(new ParameterizedTypeReference<List<User>>() {});
    }

    @Override
    public User getUserById(Long id) {
        String resourcePath = getResourcePath(id);
        logger.info("REQUEST -> GET {}", resourcePath);
        return restClient.get()
                .uri(resourcePath)
                .retrieve()
                .onStatus(HttpStatusCode::isError, (req, res) -> {
                    throw new HttpException(res.getStatusCode(), res.getHeaders());
                })
                .body(User.class);
    }

    @Override
    public User updateUser(Long id, User user) {
        String resourcePath = getResourcePath(id);
        logger.info("REQUEST -> PUT {}", resourcePath);
        return restClient.put()
                .uri(resourcePath)
                .accept(MediaType.APPLICATION_JSON)
                .body(user)
                .retrieve()
                .onStatus(HttpStatusCode::isError, (req, res) -> {
                    throw new HttpException(res.getStatusCode(), res.getHeaders());
                })
                .body(User.class);
    }

    @Override
    public void deleteUser(Long id) {
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
}
