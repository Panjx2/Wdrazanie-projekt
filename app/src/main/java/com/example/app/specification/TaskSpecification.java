package com.example.app.specification;

import com.example.app.data.Task;
import com.example.app.data.TaskStatus;
import org.springframework.data.jpa.domain.Specification;

public class TaskSpecification {
    public static Specification<Task> hasProjectId(Long projectId) {
        return (root, query, cb) ->
                projectId == null ? null :
                        cb.equal(root.get("project").get("id"), projectId);
    }
    public static Specification<Task> hasName(String name) {        return (root, query, cb) ->
            name == null ? null :
                    cb.like(cb.lower(root.get("name")), "%" + name.toLowerCase() + "%");
    }
    public static Specification<Task> hasStatus(TaskStatus status) {
        return (root, query, cb) ->
                status == null ? null :
                        cb.equal(root.get("status"), status);
    }
    public static Specification<Task> hasUserId(Long userId) {
        return (root, query, cb) ->
                userId == null ? null :
                        cb.equal(root.get("assignedUser").get("id"), userId);
    }
    public static Specification<Task> hasAssignedUsername(String username) {
        return (root, query, cb) -> {
            if (username == null || username.isEmpty()) {
                return null;
            }

            var userJoin = root.join("assignedUser", jakarta.persistence.criteria.JoinType.LEFT);

            return cb.like(
                    cb.lower(userJoin.get("username")),
                    "%" + username.toLowerCase() + "%"
            );
        };
    }
}