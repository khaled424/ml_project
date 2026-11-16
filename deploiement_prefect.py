from prefect.client.schemas.schedules import CronSchedule

from pipeline_prefect import flow_all


def main() -> None:
    """Enregistre le pipeline complet avec une exécution quotidienne."""
    flow_all.serve(
        name="ml-pipeline-all",
        schedules=[
            CronSchedule(
                cron="0 2 * * *",
                timezone="Africa/Tunis",
            )
        ],
        tags=["mlops", "full-pipeline"],
    )


if __name__ == "__main__":
    main()
